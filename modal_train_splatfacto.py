"""
Modal training script for splatfacto-da2 (masked depth prior, every2 sparse KITTI).

Quick start:
  pip install modal
  modal setup                          # one-time auth
  modal volume create kitti-nerf-data  # one-time
  modal volume create nerf-outputs     # one-time

Upload data (run once per dataset):
  modal volume put kitti-nerf-data /path/to/kitti_select_static_5seq_sparse_every2 kitti/kitti_select_static_5seq_sparse_every2
  modal volume put kitti-nerf-data /path/to/nerfstudio/kitti_seq02_0034_sparse_every2 nerfstudio/kitti_seq02_0034_sparse_every2

Train (default seq02, lambda=0.05):
  modal run modal_train_splatfacto.py::main

Override hyperparams:
  modal run modal_train_splatfacto.py::main --lambda-depth 0.1 --depth-loss-type l1 --max-num-iterations 30000

Lambda sweep (parallel GPU containers):
  modal run modal_train_splatfacto.py::sweep
  modal run modal_train_splatfacto.py::sweep --lambdas "0.0 0.05 0.1 0.2" --loss-types "mse l1"

Threshold sweep (two-stage: mask generation → parallel retrain):
  # Step 1: base train (no mask)
  modal run modal_train_splatfacto.py::main --lambda-depth 0.05
  # Step 2: sweep thresholds (generates masks then retrains, all on Modal)
  modal run modal_train_splatfacto.py::sweep_threshold \\
    --base-exp-name "kitti_seq02_0034_sparse_every2_da2_lambda0.05" \\
    --thresholds "0.08 0.12 0.16 0.22" \\
    --photo-mask-mode low

Download outputs:
  modal volume get nerf-outputs . ./local_outputs
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Paths on the local Mac (used only at image-build time)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
NERFSTUDIO_LOCAL = PROJECT_ROOT / "nerfstudio"
SCRIPTS_LOCAL = PROJECT_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Modal volumes
# ---------------------------------------------------------------------------
data_vol = modal.Volume.from_name("kitti-nerf-data", create_if_missing=True)
out_vol = modal.Volume.from_name("nerf-outputs", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/outputs"

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# Layer 1: CUDA base + system packages + PyTorch
_base = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git",
        "wget",
        "build-essential",
        "ninja-build",
        "cmake",
        "clang",
        "python3-setuptools",  # provides pkg_resources for apt Python
        "python3.10-dev",
        "libgl1",
        "ffmpeg",
    )
    .run_commands(
        # Modal's pip mirror omits pkg_resources from its setuptools wheels.
        # Copy it from the apt-installed system package so all CUDA extension
        # builds (tcnn, gsplat) can import it.
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

# Layer 2: tiny-cuda-nn (slow compile — only rebuilt when CUDA version changes)
# Target A10G (86) and A100 (80); add 90 for H100 if needed.
_with_tcnn = _base.run_commands(
    # 'wheel' is required for the legacy setup.py bdist_wheel build path used by tcnn.
    "pip install --no-cache-dir wheel setuptools",
    "MAX_JOBS=4 TCNN_CUDA_ARCHITECTURES='86;80' "
    "pip install --no-cache-dir --no-build-isolation "
    "git+https://github.com/NVlabs/tiny-cuda-nn.git"
    "@b3473c81396fe927293bdfd5a6be32df8769927c"
    "#subdirectory=bindings/torch",
)

# Layer 3: gsplat 1.4.0
_with_gsplat = _with_tcnn.run_commands(
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' MAX_JOBS=4 "
    "pip install --no-cache-dir --no-build-isolation "
    "git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0",
)

# Layer 4: nerfstudio from local source (contains the custom splatfacto-da2 model).
# Re-built when nerfstudio source files change.
image = (
    _with_gsplat
    .add_local_dir(NERFSTUDIO_LOCAL, "/opt/nerfstudio", copy=True)
    .run_commands(
        "pip install --no-cache-dir /opt/nerfstudio 'numpy<2.0.0'",
    )
    .add_local_dir(SCRIPTS_LOCAL, "/opt/project_scripts")
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = modal.App("splatfacto-da2")


def _derive_slug(kitti_seq_dir: str) -> str:
    """kitti_seq02_0034 from KITTISeq02_..._drive_0034_sync_..."""
    m = re.search(r"KITTISeq(\d+)_.*drive_(\d+)_sync", Path(kitti_seq_dir).name)
    if not m:
        raise ValueError(
            f"Cannot derive slug from: {kitti_seq_dir}\n"
            "Expected folder name like KITTISeq02_..._drive_0034_sync_..."
        )
    return f"kitti_seq{int(m.group(1)):02d}_{m.group(2)}"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 10,  # 10 hours max
    volumes={
        DATA_MOUNT: data_vol,
        OUT_MOUNT: out_vol,
    },
    # Env vars that suppress spurious CUDA / dynamo warnings
    env={"TORCHDYNAMO_DISABLE": "1", "MAX_JOBS": "1"},
)
def train(
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.05,
    depth_loss_type: str = "mse",
    depth_sup_type: str = "da2",
    max_num_iterations: int = 50000,
    photo_mask_dir: str = "",
    photo_mask_mode: str = "low",
    photo_mask_threshold: float = 0.12,
):
    import os
    import subprocess

    # ---- resolve paths -------------------------------------------------
    sparse_kitti_root = f"{DATA_MOUNT}/kitti/kitti_select_static_5seq_sparse_every2"
    seq_dir = (
        kitti_seq_dir
        if ("/" in kitti_seq_dir)
        else f"{sparse_kitti_root}/{kitti_seq_dir}"
    )

    slug = _derive_slug(seq_dir)
    nerfstudio_src = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2"
    data_dir = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2_{depth_sup_type}"
    depth_dir = f"{seq_dir}/depths_{depth_sup_type}"

    # Naming mirrors MipNeRF sweep convention:
    #   {ds_tag}_sparse_every2_{sup_type}_lambda{lambda}_{mode}{thresh_tag}_{iters}
    # e.g. kitti_seq02_0034_sparse_every2_da2_lambda0.05_low012_50000
    #      kitti_seq02_0034_sparse_every2_da2_lambda0.05_nomask_50000
    if photo_mask_dir:
        thresh_tag = f"{photo_mask_threshold:.2f}".replace(".", "")
        mask_label = f"{photo_mask_mode}{thresh_tag}"
    else:
        mask_label = "nomask"
    exp_name = f"{slug}_sparse_every2_{depth_sup_type}_lambda{lambda_depth}_{mask_label}_{max_num_iterations}"

    # ---- sanity checks -------------------------------------------------
    for path, label in [
        (seq_dir, "KITTI sequence dir"),
        (f"{nerfstudio_src}/transforms.json", "sparse nerfstudio transforms.json"),
        (depth_dir, "depth supervision folder"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {label}: {path}\n"
                "Upload data first — see the docstring at the top of modal_train.py"
            )

    if photo_mask_dir and not os.path.isdir(photo_mask_dir):
        raise FileNotFoundError(f"photo_mask_dir does not exist: {photo_mask_dir}")

    # ---- step 1: build _da2 dataset if not yet done --------------------
    if not os.path.exists(f"{data_dir}/transforms.json"):
        print(f"Building depth dataset: {data_dir}")
        subprocess.run(
            [
                "python", "/opt/project_scripts/make_nerfstudio_kitti_depth.py",
                "--src", nerfstudio_src,
                "--dst", data_dir,
                "--depth-dir", depth_dir,
                "--depth-sup-type", depth_sup_type,
                "--overwrite",
            ],
            check=True,
        )

    # ---- step 2: attach photometric masks (optional) -------------------
    if photo_mask_dir:
        print(f"Attaching photometric masks from: {photo_mask_dir}")
        subprocess.run(
            [
                "python", "/opt/project_scripts/attach_nerfstudio_photo_masks.py",
                "--data-dir", data_dir,
                "--mask-dir", photo_mask_dir,
                "--overwrite",
            ],
            check=True,
        )

    # ---- step 3: train -------------------------------------------------
    cmd = [
        "ns-train", "splatfacto-da2",
        "--data", data_dir,
        "--experiment-name", exp_name,
        "--output-dir", OUT_MOUNT,
        "--max-num-iterations", str(max_num_iterations),
        "--pipeline.model.lambda-depth", str(lambda_depth),
        "--pipeline.model.depth-loss-type", depth_loss_type,
        "--pipeline.model.photo-mask-mode", photo_mask_mode,
        "--vis", "tensorboard",
    ]

    print("=" * 60)
    print(f"Seq dir:     {seq_dir}")
    print(f"Data dir:    {data_dir}")
    print(f"Depth dir:   {depth_dir}")
    print(f"Exp name:    {exp_name}")
    print(f"Command:     {' '.join(cmd)}")
    print("=" * 60)

    subprocess.run(cmd, check=True)

    # Flush volume writes so outputs are visible after the run
    out_vol.commit()
    print(f"\nOutputs saved to Modal volume 'nerf-outputs' under: {exp_name}/")


# ---------------------------------------------------------------------------
# Local entrypoint — single run
#   modal run modal_train.py
#   modal run modal_train.py --lambda-depth 0.1 --depth-loss-type l1
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.05,
    depth_loss_type: str = "mse",
    depth_sup_type: str = "da2",
    max_num_iterations: int = 50000,
    photo_mask_dir: str = "",
    photo_mask_mode: str = "low",
):
    train.remote(
        kitti_seq_dir=kitti_seq_dir,
        lambda_depth=lambda_depth,
        depth_loss_type=depth_loss_type,
        depth_sup_type=depth_sup_type,
        max_num_iterations=max_num_iterations,
        photo_mask_dir=photo_mask_dir,
        photo_mask_mode=photo_mask_mode,
    )


# ---------------------------------------------------------------------------
# Batch sweep entrypoint — all combinations run in parallel on Modal
#
# Usage:
#   modal run modal_train.py::sweep
#   modal run modal_train.py::sweep --lambdas "0.0 0.05 0.1 0.2" --loss-types "mse l1"
#   modal run modal_train.py::sweep --seq-dirs "KITTISeq02_... KITTISeq05_..."
#
# Sweepable parameters (pass as space-separated strings):
#   --lambdas          lambda_depth values          default: "0.0 0.05 0.1 0.2"
#   --loss-types       depth_loss_type              default: "mse"
#   --seq-dirs         KITTI sequence folder names  default: seq02 only
#   --max-num-iterations  training steps            default: 50000
#   --photo-mask-dir   path on volume (optional)    default: no mask
#   --photo-mask-mode  high | low                   default: "high"
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def sweep(
    lambdas: str = "0.0 0.05 0.1 0.2",
    loss_types: str = "mse",
    seq_dirs: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    max_num_iterations: int = 50000,
    photo_mask_dir: str = "",
    photo_mask_mode: str = "low",
):
    lambda_list = [float(v) for v in lambdas.split()]
    loss_list = loss_types.split()
    seq_list = seq_dirs.split()

    configs = [
        dict(
            kitti_seq_dir=seq,
            lambda_depth=lam,
            depth_loss_type=loss,
            max_num_iterations=max_num_iterations,
            photo_mask_dir=photo_mask_dir,
            photo_mask_mode=photo_mask_mode,
        )
        for seq in seq_list
        for lam in lambda_list
        for loss in loss_list
    ]

    total = len(configs)
    print(f"\nLaunching {total} parallel runs:")
    for i, c in enumerate(configs):
        print(f"  [{i+1}/{total}] seq={Path(c['kitti_seq_dir']).name[:20]}...  "
              f"lambda={c['lambda_depth']}  loss={c['depth_loss_type']}")

    # Modal 1.x starmap does *item, so dicts expand to keys — use tuples instead.
    # Order matches train(): kitti_seq_dir, lambda_depth, depth_loss_type, depth_sup_type,
    #                        max_num_iterations, photo_mask_dir, photo_mask_mode, photo_mask_threshold
    for result in train.starmap([
        (c['kitti_seq_dir'], c['lambda_depth'], c['depth_loss_type'],
         "da2", c['max_num_iterations'], c['photo_mask_dir'],
         c['photo_mask_mode'], 0.12)
        for c in configs
    ]):
        pass


# ---------------------------------------------------------------------------
# generate_masks — render a trained checkpoint and threshold the L1 RGB error
#
# Writes mask PNGs to DATA_MOUNT/masks/<exp>_<mode><thresh_tag>/
# Returns the mask dir path (str) so sweep_threshold can chain into training.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={
        DATA_MOUNT: data_vol,
        OUT_MOUNT: out_vol,
    },
    env={"TORCHDYNAMO_DISABLE": "1"},
)
def generate_masks(
    base_exp_name: str,
    threshold: float,
    photo_mask_mode: str = "low",
    load_step: int = -1,
) -> str:
    import glob
    import subprocess

    # Find config.yml produced by the base training run
    pattern = f"{OUT_MOUNT}/{base_exp_name}/splatfacto-da2/*/config.yml"
    configs = sorted(glob.glob(pattern))
    if not configs:
        raise FileNotFoundError(
            f"No config.yml found matching:\n  {pattern}\n"
            "Run the base training first: modal run modal_train.py"
        )
    config_path = configs[-1]  # latest timestamp wins
    print(f"Using config: {config_path}")

    # Mask dir on data volume: masks/<exp>_<mode><thresh_tag>/
    thresh_tag = f"{threshold:.2f}".replace(".", "")  # 0.12 -> "012"
    mask_dir = f"{DATA_MOUNT}/masks/{base_exp_name}_{photo_mask_mode}{thresh_tag}"

    cmd = [
        "python", "/opt/project_scripts/generate_splatfacto_photo_masks.py",
        "--load-config", config_path,
        "--output-dir", mask_dir,
        "--threshold", str(threshold),
        "--photo-mask-mode", photo_mask_mode,
    ]
    if load_step >= 0:
        cmd += ["--load-step", str(load_step)]

    print(f"Generating masks → {mask_dir}")
    subprocess.run(cmd, check=True)
    data_vol.commit()
    print(f"Masks written to: {mask_dir}")
    return mask_dir


# ---------------------------------------------------------------------------
# sweep_threshold — full two-stage pipeline on Modal:
#   Stage 1: generate masks at all thresholds in parallel (GPU render)
#   Stage 2: retrain with each mask set in parallel (GPU train)
#
# Usage:
#   modal run modal_train.py::sweep_threshold \
#     --base-exp-name "kitti_seq02_0034_sparse_every2_da2_lambda0.05" \
#     --thresholds "0.08 0.12 0.16 0.22" \
#     --photo-mask-mode low \
#     --lambda-depth 0.05
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def sweep_threshold(
    base_exp_name: str,
    thresholds: str = "0.08 0.12 0.16 0.22",
    photo_mask_mode: str = "low",
    # training params carried over into the masked retrains
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.05,
    depth_loss_type: str = "mse",
    max_num_iterations: int = 50000,
):
    threshold_list = [float(t) for t in thresholds.split()]

    print(f"\nBase experiment: {base_exp_name}")
    print(f"Thresholds:      {threshold_list}")
    print(f"Mask mode:       {photo_mask_mode}")
    print(f"Retraining with: lambda={lambda_depth}  loss={depth_loss_type}  iters={max_num_iterations}")

    # ---- Stage 1: generate masks for all thresholds in parallel ----------
    print(f"\n[Stage 1] Generating masks for {len(threshold_list)} thresholds in parallel...")
    mask_gen_configs = [
        dict(
            base_exp_name=base_exp_name,
            threshold=t,
            photo_mask_mode=photo_mask_mode,
        )
        for t in threshold_list
    ]
    mask_dirs = list(generate_masks.starmap(mask_gen_configs))

    print("\n[Stage 1] Done. Mask directories:")
    for d in mask_dirs:
        print(f"  {d}")

    # ---- Stage 2: retrain with each mask set in parallel -----------------
    print(f"\n[Stage 2] Launching {len(mask_dirs)} masked retrains in parallel...")
    train_configs = [
        dict(
            kitti_seq_dir=kitti_seq_dir,
            lambda_depth=lambda_depth,
            depth_loss_type=depth_loss_type,
            max_num_iterations=max_num_iterations,
            photo_mask_dir=mask_dir,
            photo_mask_mode=photo_mask_mode,
            photo_mask_threshold=threshold,
        )
        for mask_dir, threshold in zip(mask_dirs, threshold_list)
    ]
    for _ in train.starmap(train_configs):
        pass

    print("\n[Stage 2] All threshold sweep runs complete.")


# ---------------------------------------------------------------------------
# sweep_lambda_threshold — sweep both lambda × threshold in parallel
#
# Assumes nomask base runs already exist for each lambda (run sweep first).
#
# Usage:
#   modal run modal_train_splatfacto.py::sweep_lambda_threshold \
#     --lambdas "0.0 0.1 0.15" \
#     --thresholds "0.22" \
#     --photo-mask-mode low
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def sweep_lambda_threshold(
    lambdas: str = "0.0 0.1 0.15",
    thresholds: str = "0.22",
    photo_mask_mode: str = "low",
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    depth_loss_type: str = "mse",
    depth_sup_type: str = "da2",
    max_num_iterations: int = 50000,
):
    lambda_list = [float(v) for v in lambdas.split()]
    threshold_list = [float(t) for t in thresholds.split()]
    slug = _derive_slug(kitti_seq_dir)

    # All (lambda, threshold) combinations
    combos = [(lam, t) for lam in lambda_list for t in threshold_list]

    print(f"\nLambdas:    {lambda_list}")
    print(f"Thresholds: {threshold_list}")
    print(f"Total runs: {len(combos)} mask-gen + {len(combos)} retrain")

    # ---- Stage 1: generate masks for every (lambda, threshold) in parallel --
    # Each lambda has its own nomask base checkpoint.
    print(f"\n[Stage 1] Generating {len(combos)} mask sets in parallel...")
    base_exp_names = [
        f"{slug}_sparse_every2_{depth_sup_type}_lambda{lam}_nomask_{max_num_iterations}"
        for lam, _ in combos
    ]
    for i, (name, (lam, t)) in enumerate(zip(base_exp_names, combos)):
        print(f"  [{i+1}/{len(combos)}] base={name}  threshold={t}")

    mask_dirs = list(generate_masks.starmap([
        (base_exp, t, photo_mask_mode)
        for base_exp, (_, t) in zip(base_exp_names, combos)
    ]))

    print("\n[Stage 1] Done. Mask directories:")
    for d in mask_dirs:
        print(f"  {d}")

    # ---- Stage 2: retrain all combinations in parallel ----------------------
    print(f"\n[Stage 2] Launching {len(combos)} masked retrains in parallel...")
    for result in train.starmap([
        (kitti_seq_dir, lam, depth_loss_type, depth_sup_type,
         max_num_iterations, mask_dir, photo_mask_mode, t)
        for (lam, t), mask_dir in zip(combos, mask_dirs)
    ]):
        pass

    print("\n[Stage 2] All lambda × threshold sweep runs complete.")


# ---------------------------------------------------------------------------
# eval — run ns-eval on a completed training run and print final metrics
#
# Usage:
#   modal run modal_train_splatfacto.py::run_eval \
#     --exp-name "kitti_seq02_0034_sparse_every2_da2_lambda0.05_nomask_50000"
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 1,
    volumes={
        DATA_MOUNT: data_vol,
        OUT_MOUNT: out_vol,
    },
    env={"TORCHDYNAMO_DISABLE": "1"},
)
def eval_run(exp_name: str) -> dict:
    import glob, json, subprocess

    pattern = f"{OUT_MOUNT}/{exp_name}/splatfacto-da2/*/config.yml"
    configs = sorted(glob.glob(pattern))
    if not configs:
        raise FileNotFoundError(f"No config.yml found: {pattern}")
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
        metrics = json.load(f)
    return metrics


@app.local_entrypoint()
def run_eval(
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.05,
    depth_loss_type: str = "mse",
    depth_sup_type: str = "da2",
    max_num_iterations: int = 50000,
    photo_mask_dir: str = "",
    photo_mask_mode: str = "low",
    photo_mask_threshold: float = 0.12,
):
    import json

    slug = _derive_slug(kitti_seq_dir)
    if photo_mask_dir:
        thresh_tag = f"{photo_mask_threshold:.2f}".replace(".", "")
        mask_label = f"{photo_mask_mode}{thresh_tag}"
    else:
        mask_label = "nomask"
    exp_name = f"{slug}_sparse_every2_{depth_sup_type}_lambda{lambda_depth}_{mask_label}_{max_num_iterations}"

    print(f"Evaluating: {exp_name}")
    metrics = eval_run.remote(exp_name=exp_name)
    print("\n========== Eval Results ==========")
    print(json.dumps(metrics, indent=2))
    print("==")
