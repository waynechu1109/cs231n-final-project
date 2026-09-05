"""Modal training script for splatfacto-da2 (masked depth prior on sparse KITTI).

Setup, dataset upload, entrypoints, and cost estimates are documented in
docs/modal-splatfacto.md.
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
DA2_LOCAL = PROJECT_ROOT / "Depth-Anything-V2"

# ---------------------------------------------------------------------------
# Modal volumes
# ---------------------------------------------------------------------------
data_vol = modal.Volume.from_name("kitti-nerf-data", create_if_missing=True)
out_vol = modal.Volume.from_name("nerf-outputs", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/outputs"

# Mip-NeRF 360 originals are ~5k px; splatfacto-da2 uses full-image eval + SSIM on GPU.
# Factor 4 matches the Kaggle pack's images_4/ and keeps A10G (~24GB) under budget.
MIP360_DOWNSCALE_FACTOR = 4

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

# Lightweight image for DA2 depth inference (no tcnn/gsplat needed).
_da2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-runtime-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2+cu118",
        "torchvision==0.16.2+cu118",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu118",
    )
    .pip_install("numpy<2.0.0", "opencv-python-headless", "huggingface_hub", "tqdm")
    .add_local_dir(DA2_LOCAL, "/opt/da2", copy=True)
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


def _resolve_dataset_paths(
    dataset_family: str,
    kitti_seq_dir: str,
    mip360_scene: str,
    depth_sup_type: str,
) -> tuple[str, str, str, str, str]:
    """Return (seq_dir, slug, nerfstudio_src, data_dir, sparse_tag)."""
    if dataset_family == "mip360":
        scene = mip360_scene
        seq_dir = (
            kitti_seq_dir
            if "/" in kitti_seq_dir
            else f"{DATA_MOUNT}/mip360_sparse/{scene}"
        )
        slug = f"{scene}_sparse"
        nerfstudio_src = f"{DATA_MOUNT}/nerfstudio/{slug}"
        data_dir = f"{DATA_MOUNT}/nerfstudio/{slug}_{depth_sup_type}"
        sparse_tag = ""
    elif dataset_family == "kitti":
        sparse_kitti_root = f"{DATA_MOUNT}/kitti/kitti_select_static_5seq_sparse_every2"
        seq_dir = (
            kitti_seq_dir
            if "/" in kitti_seq_dir
            else f"{sparse_kitti_root}/{kitti_seq_dir}"
        )
        slug = _derive_slug(seq_dir)
        nerfstudio_src = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2"
        data_dir = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2_{depth_sup_type}"
        sparse_tag = "_sparse_every2"
    else:
        raise ValueError(f"Unknown dataset_family: {dataset_family}")
    return seq_dir, slug, nerfstudio_src, data_dir, sparse_tag


def _experiment_name(
    slug: str,
    sparse_tag: str,
    depth_sup_type: str,
    lambda_depth: float,
    max_num_iterations: int,
    *,
    photo_mask_dir: str = "",
    photo_mask_mode: str = "low",
    photo_mask_threshold: float = 0.12,
    masked: bool = False,
    mask_label: str = "",
) -> str:
    if mask_label:
        final_label = mask_label
    elif photo_mask_dir or masked:
        thresh_tag = f"{photo_mask_threshold:.2f}".replace(".", "")
        final_label = f"{photo_mask_mode}{thresh_tag}"
    else:
        final_label = "nomask"
    return (
        f"{slug}{sparse_tag}_{depth_sup_type}_lambda{lambda_depth}_"
        f"{final_label}_{max_num_iterations}"
    )


def _wire_mip360_downscale_links(
    data_dir: str,
    seq_dir: str,
    depth_sup_type: str,
    *,
    commit_volume: bool = False,
) -> None:
    """Nerfstudio maps RGB/depth/masks to {images,depths,photo_masks}_{factor}/ when downscaling."""
    import os

    images_n = f"{seq_dir}/images_{MIP360_DOWNSCALE_FACTOR}"
    images_n_link = f"{data_dir}/images_{MIP360_DOWNSCALE_FACTOR}"
    if not os.path.isdir(images_n):
        raise FileNotFoundError(
            f"Missing Mip-360 downscaled RGB folder: {images_n}\n"
            "Upload mip360_sparse/<scene>/images_4 from the 360_v2 pack."
        )
    if not os.path.exists(images_n_link):
        os.symlink(images_n, images_n_link)

    depths_sup_link = f"{data_dir}/depths_{depth_sup_type}"
    depths_n_link = f"{data_dir}/depths_{MIP360_DOWNSCALE_FACTOR}"
    if not (os.path.isdir(depths_sup_link) or os.path.islink(depths_sup_link)):
        raise FileNotFoundError(
            f"Missing depth symlink in dataset: {depths_sup_link}\n"
            f"Rebuild depth dataset under {data_dir} (delete stale _da2 and re-run train)."
        )
    if not os.path.exists(depths_n_link):
        os.symlink(f"depths_{depth_sup_type}", depths_n_link)

    photo_masks_link = f"{data_dir}/photo_masks"
    photo_masks_n_link = f"{data_dir}/photo_masks_{MIP360_DOWNSCALE_FACTOR}"
    if (os.path.isdir(photo_masks_link) or os.path.islink(photo_masks_link)) and not os.path.exists(
        photo_masks_n_link
    ):
        os.symlink("photo_masks", photo_masks_n_link)

    if commit_volume:
        data_vol.commit()


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
    dataset_family: str = "kitti",
    mip360_scene: str = "bicycle",
    mask_label: str = "",
    seed: int = 0,
):
    import os
    import shutil
    import subprocess

    seq_dir, slug, nerfstudio_src, data_dir, sparse_tag = _resolve_dataset_paths(
        dataset_family, kitti_seq_dir, mip360_scene, depth_sup_type
    )
    depth_dir = f"{seq_dir}/depths_{depth_sup_type}"

    effective_mask_label = mask_label
    if seed > 0:
        base_label = mask_label or ("nomask" if not (photo_mask_dir or False) else "")
        effective_mask_label = f"{base_label}_seed{seed}" if base_label else f"seed{seed}"
    exp_name = _experiment_name(
        slug, sparse_tag, depth_sup_type, lambda_depth, max_num_iterations,
        photo_mask_dir=photo_mask_dir,
        photo_mask_mode=photo_mask_mode,
        photo_mask_threshold=photo_mask_threshold,
        mask_label=effective_mask_label,
    )

    # ---- sanity checks -------------------------------------------------
    dataset_already_built = os.path.exists(f"{data_dir}/transforms.json")
    checks = [
        (seq_dir, "KITTI sequence dir"),
        (f"{nerfstudio_src}/transforms.json", "sparse nerfstudio transforms.json"),
    ]
    # Skip depth_dir check when the nerfstudio dataset is already built — depths
    # are embedded in data_dir and may no longer exist at the raw seq_dir path.
    if not dataset_already_built:
        checks.append((depth_dir, "depth supervision folder"))
    for path, label in checks:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {label}: {path}\n"
                "Upload data first — see the docstring at the top of modal_train.py"
            )

    if photo_mask_dir and not os.path.isdir(photo_mask_dir):
        raise FileNotFoundError(f"photo_mask_dir does not exist: {photo_mask_dir}")

    # ---- step 1: build _da2 dataset if not yet done --------------------
    if not os.path.exists(f"{data_dir}/transforms.json"):
        print(f"Building depth dataset: {data_dir} ({dataset_family})")
        make_cmd = [
            "python", "/opt/project_scripts/data_prep/make_nerfstudio_kitti_depth.py",
            "--src", nerfstudio_src,
            "--dst", data_dir,
            "--depth-dir", depth_dir,
            "--depth-sup-type", depth_sup_type,
            "--overwrite",
        ]
        # Mip-360: transforms.json only on volume; RGB + depth live under mip360_sparse/.
        if dataset_family == "mip360":
            make_cmd += [
                "--images-dir", f"{seq_dir}/images",
                "--skip-missing-frames",
            ]
        subprocess.run(make_cmd, check=True)
        data_vol.commit()

    # Strip stale mask metadata only after _da2 exists.
    da2_shared = f"{DATA_MOUNT}/nerfstudio/{slug}{sparse_tag}_{depth_sup_type}"
    if os.path.exists(f"{da2_shared}/transforms.json"):
        subprocess.run(
            [
                "python", "/opt/project_scripts/masks/attach_nerfstudio_photo_masks.py",
                "--data-dir", da2_shared,
                "--strip-only",
            ],
            check=True,
        )
        data_vol.commit()

    # ---- step 2: attach photometric masks (optional) -------------------
    if photo_mask_dir:
        # Parallel masked runs must not share one transforms.json / photo_masks link.
        masked_data_dir = f"{data_dir}__{exp_name}"
        if not os.path.exists(masked_data_dir):
            print(f"Copying dataset for masked training: {masked_data_dir}")
            shutil.copytree(data_dir, masked_data_dir, symlinks=True)
        data_dir = masked_data_dir
        print(f"Attaching photometric masks from: {photo_mask_dir}")
        subprocess.run(
            [
                "python", "/opt/project_scripts/masks/attach_nerfstudio_photo_masks.py",
                "--data-dir", data_dir,
                "--mask-dir", photo_mask_dir,
                "--overwrite",
            ],
            check=True,
        )
        data_vol.commit()

    if dataset_family == "mip360":
        _wire_mip360_downscale_links(
            data_dir, seq_dir, depth_sup_type, commit_volume=True
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
        "--machine.seed", str(seed),
        "--vis", "tensorboard",
    ]
    # Dataparser flags use the nerfstudio-data subcommand (not --pipeline.datamanager...).
    if dataset_family == "mip360":
        cmd += [
            "nerfstudio-data",
            "--downscale-factor",
            str(MIP360_DOWNSCALE_FACTOR),
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
    dataset_family: str = "kitti",
    mip360_scene: str = "bicycle",
):
    train.remote(
        kitti_seq_dir=kitti_seq_dir,
        lambda_depth=lambda_depth,
        depth_loss_type=depth_loss_type,
        depth_sup_type=depth_sup_type,
        max_num_iterations=max_num_iterations,
        photo_mask_dir=photo_mask_dir,
        photo_mask_mode=photo_mask_mode,
        dataset_family=dataset_family,
        mip360_scene=mip360_scene,
    )


# ---------------------------------------------------------------------------
# run_headline_seeds — rerun the headline Splatfacto setting (τ=0.18, λ=0.10)
# with additional seeds for variance estimation.  Seed 0 (the original run)
# already exists; this launches seeds 1 and 2 in parallel.
#
# Usage:
#   modal run modal_train_splatfacto.py::run_headline_seeds
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def run_headline_seeds(
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.1,
    photo_mask_threshold: float = 0.18,
    max_num_iterations: int = 50000,
    seeds: str = "1,2",
):
    """Run the headline Splatfacto setting over extra seeds in parallel.

    The τ=0.18, λ=0.10 mask already exists on the Modal data volume at
    masks/kitti_seq02_0034_sparse_every2_da2_lambda0.1_nomask_50000_low018.
    Seed 0 is the original published run; seeds 1,2 are new.
    """
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    thresh_tag = f"{photo_mask_threshold:.2f}".replace(".", "")
    mask_dir = (
        f"{DATA_MOUNT}/masks/"
        f"kitti_seq02_0034_sparse_every2_da2_lambda{lambda_depth}_nomask_{max_num_iterations}"
        f"_low{thresh_tag}"
    )

    print(f"Headline mask dir: {mask_dir}")
    print(f"Launching seeds: {seed_list}")

    # Tuple order: kitti_seq_dir, lambda_depth, depth_loss_type, depth_sup_type,
    #   max_num_iterations, photo_mask_dir, photo_mask_mode, photo_mask_threshold,
    #   dataset_family, mip360_scene, mask_label, seed
    tuples = [
        (
            kitti_seq_dir, lambda_depth, "mse", "da2",
            max_num_iterations, mask_dir, "low", photo_mask_threshold,
            "kitti", "bicycle", f"low{thresh_tag}", s,
        )
        for s in seed_list
    ]
    for _ in train.starmap(tuples):
        pass
    print("All headline seed runs complete.")


# ---------------------------------------------------------------------------
# run_global_seeds — run global (unmasked, τ=1.00) supervision at λ=0.10
# with seeds 0, 1, 2 in parallel for a proper paired comparison against the
# masked headline runs.
#
# Seed 0 may already exist as the original global run; set --seeds to skip it.
#
# Usage:
#   modal run modal_train_splatfacto.py::run_global_seeds
#   modal run modal_train_splatfacto.py::run_global_seeds --seeds "1,2"
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def run_global_seeds(
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.1,
    photo_mask_threshold: float = 1.0,
    max_num_iterations: int = 50000,
    seeds: str = "0,1,2",
):
    """Run global (τ=1.00) Splatfacto over multiple seeds in parallel.

    Uses the same mask dir structure as run_headline_seeds but with threshold=1.0,
    so the mask keeps every pixel (equivalent to unmasked depth supervision).
    Seeds match those used in run_headline_seeds for paired comparison.
    """
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    thresh_tag = f"{photo_mask_threshold:.2f}".replace(".", "")  # 1.00 -> "100"
    mask_dir = (
        f"{DATA_MOUNT}/masks/"
        f"kitti_seq02_0034_sparse_every2_da2_lambda{lambda_depth}_nomask_{max_num_iterations}"
        f"_low{thresh_tag}"
    )

    print(f"Global mask dir: {mask_dir}")
    print(f"Launching seeds: {seed_list}")

    tuples = [
        (
            kitti_seq_dir, lambda_depth, "mse", "da2",
            max_num_iterations, mask_dir, "low", photo_mask_threshold,
            "kitti", "bicycle", f"low{thresh_tag}", s,
        )
        for s in seed_list
    ]
    for _ in train.starmap(tuples):
        pass
    print("All global seed runs complete.")


# ---------------------------------------------------------------------------
# run_extra_seeds — run additional seeds for ANY sequence, masked or nomask.
#
# Usage (masked τ=0.18 seeds 1+2 for Seq00):
#   modal run modal_train_splatfacto.py::run_extra_seeds \
#     --kitti-seq-dir "KITTISeq00_2011_10_03_drive_0027_sync_llffdtu_s2700_e3000_densegt" \
#     --condition masked --seeds "1,2"
#
# Usage (global/nomask seeds 1+2 for Seq05):
#   modal run modal_train_splatfacto.py::run_extra_seeds \
#     --kitti-seq-dir "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt" \
#     --condition nomask --seeds "1,2"
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def run_extra_seeds(
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    condition: str = "masked",   # "masked" (τ=0.18) or "nomask" (global)
    lambda_depth: float = 0.1,
    photo_mask_threshold: float = 0.18,
    max_num_iterations: int = 50000,
    seeds: str = "1,2",
):
    """Run additional seeds for any KITTI sequence in masked or nomask condition.

    condition="masked": uses the pre-computed τ=0.18 photometric mask for the sequence.
    condition="nomask":  no photometric mask (global depth supervision on all valid pixels).
    """
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    slug = _derive_slug(kitti_seq_dir)

    if condition == "masked":
        thresh_tag = f"{photo_mask_threshold:.2f}".replace(".", "")
        mask_dir = (
            f"{DATA_MOUNT}/masks/"
            f"{slug}_sparse_every2_da2_lambda{lambda_depth}_nomask_{max_num_iterations}"
            f"_low{thresh_tag}"
        )
        mask_label = f"low{thresh_tag}"
        threshold = photo_mask_threshold
        mask_mode = "low"
    else:  # nomask / global
        mask_dir = ""
        mask_label = "nomask"
        threshold = 0.18   # unused when no mask dir
        mask_mode = "low"

    print(f"Seq: {slug}  condition: {condition}")
    print(f"mask_dir: {mask_dir or '(none)'}")
    print(f"Launching seeds: {seed_list}")

    tuples = [
        (
            kitti_seq_dir, lambda_depth, "mse", "da2",
            max_num_iterations, mask_dir, mask_mode, threshold,
            "kitti", "bicycle", mask_label, s,
        )
        for s in seed_list
    ]
    for _ in train.starmap(tuples):
        pass
    print(f"All {condition} seed runs complete for {slug}.")


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
    dataset_family: str = "kitti",
    mip360_scene: str = "bicycle",
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
            dataset_family=dataset_family,
            mip360_scene=mip360_scene,
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
        (
            c["kitti_seq_dir"],
            c["lambda_depth"],
            c["depth_loss_type"],
            "da2",
            c["max_num_iterations"],
            c["photo_mask_dir"],
            c["photo_mask_mode"],
            0.12,
            c["dataset_family"],
            c["mip360_scene"],
        )
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

    # Derive data_dir from base_exp_name (strip _lambda... suffix)
    data_dir_name = base_exp_name.split("_lambda")[0]
    data_dir = f"{DATA_MOUNT}/nerfstudio/{data_dir_name}"

    cmd = [
        "python", "/opt/project_scripts/masks/generate_splatfacto_photo_masks.py",
        "--load-config", config_path,
        "--output-dir", mask_dir,
        "--threshold", str(threshold),
        "--photo-mask-mode", photo_mask_mode,
        "--data-dir", data_dir,
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
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.05,
    depth_loss_type: str = "mse",
    max_num_iterations: int = 50000,
    dataset_family: str = "kitti",
    mip360_scene: str = "bicycle",
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
    for _ in train.starmap([
        (
            kitti_seq_dir,
            lambda_depth,
            depth_loss_type,
            "da2",
            max_num_iterations,
            mask_dir,
            photo_mask_mode,
            threshold,
            dataset_family,
            mip360_scene,
        )
        for mask_dir, threshold in zip(mask_dirs, threshold_list)
    ]):
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
    dataset_family: str = "kitti",
    mip360_scene: str = "bicycle",
):
    lambda_list = [float(v) for v in lambdas.split()]
    threshold_list = [float(t) for t in thresholds.split()]
    _, slug, _, _, sparse_tag = _resolve_dataset_paths(
        dataset_family, kitti_seq_dir, mip360_scene, depth_sup_type
    )

    # All (lambda, threshold) combinations
    combos = [(lam, t) for lam in lambda_list for t in threshold_list]

    print(f"\nLambdas:    {lambda_list}")
    print(f"Thresholds: {threshold_list}")
    print(f"Total runs: {len(combos)} mask-gen + {len(combos)} retrain")

    # ---- Stage 1: generate masks for every (lambda, threshold) in parallel --
    # Each lambda has its own nomask base checkpoint.
    print(f"\n[Stage 1] Generating {len(combos)} mask sets in parallel...")
    base_exp_names = [
        f"{slug}{sparse_tag}_{depth_sup_type}_lambda{lam}_nomask_{max_num_iterations}"
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
        (
            kitti_seq_dir,
            lam,
            depth_loss_type,
            depth_sup_type,
            max_num_iterations,
            mask_dir,
            photo_mask_mode,
            t,
            dataset_family,
            mip360_scene,
        )
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
    masked: bool = False,
    dataset_family: str = "kitti",
    mip360_scene: str = "bicycle",
    mask_label: str = "",
):
    import json

    _, slug, _, _, sparse_tag = _resolve_dataset_paths(
        dataset_family, kitti_seq_dir, mip360_scene, depth_sup_type
    )
    exp_name = _experiment_name(
        slug, sparse_tag, depth_sup_type, lambda_depth, max_num_iterations,
        photo_mask_dir=photo_mask_dir,
        photo_mask_mode=photo_mask_mode,
        photo_mask_threshold=photo_mask_threshold,
        masked=masked,
        mask_label=mask_label,
    )

    print(f"Evaluating: {exp_name}")
    metrics = eval_run.remote(exp_name=exp_name)
    print("\n========== Eval Results ==========")
    print(json.dumps(metrics, indent=2))
    print("==")


# ---------------------------------------------------------------------------
# eval_by_name — eval a run by its exact experiment directory name
#
# Usage:
#   modal run modal_train_splatfacto.py::eval_by_name \
#     --exp-name "kitti_seq02_0034_sparse_every2_da2_lambda0.1_low100_seed1_50000"
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def eval_by_name(exp_name: str):
    import json
    print(f"Evaluating: {exp_name}")
    metrics = eval_run.remote(exp_name=exp_name)
    print("\n========== Eval Results ==========")
    print(json.dumps(metrics, indent=2))
    print("==================================")


# ---------------------------------------------------------------------------
# sweep_eval — eval all lambda values in parallel
#
# Usage (bicycle masked @ 0.14):
#   modal run modal_train_splatfacto.py::sweep_eval \
#     --dataset-family mip360 --mip360-scene bicycle \
#     --lambdas "0.0 0.05 0.1 0.15" \
#     --photo-mask-mode low --photo-mask-threshold 0.14 --masked
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def sweep_eval(
    lambdas: str = "0.0 0.05 0.1 0.15",
    depth_sup_type: str = "da2",
    max_num_iterations: int = 50000,
    photo_mask_mode: str = "low",
    photo_mask_threshold: float = 0.14,
    masked: bool = True,
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    dataset_family: str = "kitti",
    mip360_scene: str = "bicycle",
):
    import json

    lambda_list = [float(v) for v in lambdas.split()]
    _, slug, _, _, sparse_tag = _resolve_dataset_paths(
        dataset_family, kitti_seq_dir, mip360_scene, depth_sup_type
    )
    exp_names = [
        _experiment_name(
            slug, sparse_tag, depth_sup_type, lam, max_num_iterations,
            photo_mask_mode=photo_mask_mode,
            photo_mask_threshold=photo_mask_threshold,
            masked=masked,
        )
        for lam in lambda_list
    ]

    print(f"\nEvaluating {len(exp_names)} experiments in parallel:")
    for name in exp_names:
        print(f"  {name}")

    results = list(eval_run.map(exp_names))

    print("\n========== Eval Summary ==========")
    summary = {}
    for lam, name, metrics in zip(lambda_list, exp_names, results):
        summary[f"lambda{lam}"] = {"exp_name": name, **metrics}
        print(f"\n--- lambda={lam}  ({name}) ---")
        print(json.dumps(metrics, indent=2))
    print("\n========== Combined ==========")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# generate_matched_masks — matched-ratio mask generation for ablation study
#
# Generates masks that select exactly as many valid-depth pixels as low018,
# but using a different selection criterion (high-error or random).
#
# Usage:
#   modal run modal_train_splatfacto.py::run_matched_ablation \
#     --base-exp-name "kitti_seq02_0034_sparse_every2_da2_lambda0.1_nomask_50000" \
#     --lambda-depth 0.1
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
def generate_matched_masks(
    base_exp_name: str,
    mode: str,
    ref_threshold: float = 0.18,
    seed: int = 0,
) -> tuple:
    """Generate matched-ratio masks and return (mask_dir, mask_label).

    mode must be "high_matched" or "random_matched".
    Masks are stored at DATA_MOUNT/masks/<base_exp_name>_<mask_label>/
    """
    import glob
    import subprocess

    pattern = f"{OUT_MOUNT}/{base_exp_name}/splatfacto-da2/*/config.yml"
    configs = sorted(glob.glob(pattern))
    if not configs:
        raise FileNotFoundError(
            f"No config.yml found matching:\n  {pattern}\n"
            "Run the nomask base training first."
        )
    config_path = configs[-1]
    print(f"Using config: {config_path}")

    # Build mask label and output dir
    ref_tag = f"{ref_threshold:.2f}".replace(".", "")  # 0.18 -> "018"
    if mode == "high_matched":
        mask_label = f"high_error_matched_low{ref_tag}"
    elif mode == "random_matched":
        mask_label = f"random_matched_low{ref_tag}_seed{seed}"
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'high_matched' or 'random_matched'.")

    mask_dir = f"{DATA_MOUNT}/masks/{base_exp_name}_{mask_label}"

    cmd = [
        "python", "/opt/project_scripts/masks/generate_matched_ratio_masks.py",
        "--load-config", config_path,
        "--output-dir", mask_dir,
        "--mode", mode,
        "--ref-threshold", str(ref_threshold),
        "--seed", str(seed),
    ]
    print(f"Generating {mode} masks → {mask_dir}")
    subprocess.run(cmd, check=True)
    data_vol.commit()
    print(f"Masks written to: {mask_dir}  label: {mask_label}")
    return mask_dir, mask_label


@app.local_entrypoint()
def run_matched_ablation(
    base_exp_name: str = "kitti_seq02_0034_sparse_every2_da2_lambda0.1_nomask_50000",
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.1,
    ref_threshold: float = 0.18,
    seed: int = 0,
    depth_loss_type: str = "mse",
    max_num_iterations: int = 50000,
):
    """Two-stage ablation pipeline.

    Stage 1: generate high_error_matched and random_matched masks in parallel.
    Stage 2: train both masked experiments in parallel.

    Experiment names produced:
      kitti_seq02_0034_sparse_every2_da2_lambda<l>_high_error_matched_low018_<iters>
      kitti_seq02_0034_sparse_every2_da2_lambda<l>_random_matched_low018_seed0_<iters>
    """
    import json

    print(f"\nBase experiment: {base_exp_name}")
    print(f"lambda_depth: {lambda_depth}  ref_threshold: {ref_threshold}  seed: {seed}")

    # Stage 1: generate both mask sets in parallel
    print("\n[Stage 1] Generating matched-ratio masks (high_error + random) in parallel...")
    mask_results = list(generate_matched_masks.starmap([
        (base_exp_name, "high_matched", ref_threshold, seed),
        (base_exp_name, "random_matched", ref_threshold, seed),
    ]))
    (high_mask_dir, high_label), (rand_mask_dir, rand_label) = mask_results
    print(f"\n  high_matched  mask dir:   {high_mask_dir}")
    print(f"  high_matched  mask label: {high_label}")
    print(f"  random_matched mask dir:   {rand_mask_dir}")
    print(f"  random_matched mask label: {rand_label}")

    # Stage 2: train both experiments in parallel
    # Tuple order matches train() positional signature:
    #   kitti_seq_dir, lambda_depth, depth_loss_type, depth_sup_type,
    #   max_num_iterations, photo_mask_dir, photo_mask_mode, photo_mask_threshold,
    #   dataset_family, mip360_scene, mask_label
    print("\n[Stage 2] Launching 2 masked training runs in parallel...")
    for _ in train.starmap([
        (
            kitti_seq_dir, lambda_depth, depth_loss_type, "da2",
            max_num_iterations, high_mask_dir, "low", ref_threshold,
            "kitti", "bicycle", high_label,
        ),
        (
            kitti_seq_dir, lambda_depth, depth_loss_type, "da2",
            max_num_iterations, rand_mask_dir, "low", ref_threshold,
            "kitti", "bicycle", rand_label,
        ),
    ]):
        pass

    print("\n========== Matched-ratio ablation complete ==========")
    print(f"Evaluate with:")
    print(f"  modal run modal_train_splatfacto.py::run_eval \\")
    print(f"    --lambda-depth {lambda_depth} --mask-label '{high_label}'")
    print(f"  modal run modal_train_splatfacto.py::run_eval \\")
    print(f"    --lambda-depth {lambda_depth} --mask-label '{rand_label}'")


@app.local_entrypoint()
def run_random_seed_sweep(
    base_exp_name: str = "kitti_seq02_0034_sparse_every2_da2_lambda0.1_nomask_50000",
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.1,
    ref_threshold: float = 0.18,
    seeds: str = "1,2",
    depth_loss_type: str = "mse",
    max_num_iterations: int = 50000,
):
    """Generate random_matched masks and train for multiple seeds in parallel.

    Use this to add extra seeds to an existing random_matched ablation
    (seed=0 already trained via run_matched_ablation).

    Example:
      modal run modal_train_splatfacto.py::run_random_seed_sweep --seeds "1,2"
    """
    seed_list = [int(s.strip()) for s in seeds.split(",")]
    print(f"\nBase experiment: {base_exp_name}")
    print(f"lambda_depth: {lambda_depth}  ref_threshold: {ref_threshold}  seeds: {seed_list}")

    # Stage 1: generate random_matched masks for all seeds in parallel
    print(f"\n[Stage 1] Generating random_matched masks for seeds {seed_list} in parallel...")
    mask_args = [(base_exp_name, "random_matched", ref_threshold, s) for s in seed_list]
    mask_results = list(generate_matched_masks.starmap(mask_args))

    for (mask_dir, mask_label) in mask_results:
        print(f"  {mask_label}: {mask_dir}")

    # Stage 2: train all seeds in parallel
    print(f"\n[Stage 2] Launching {len(seed_list)} training runs in parallel...")
    train_args = [
        (
            kitti_seq_dir, lambda_depth, depth_loss_type, "da2",
            max_num_iterations, mask_dir, "low", ref_threshold,
            "kitti", "bicycle", mask_label,
        )
        for (mask_dir, mask_label) in mask_results
    ]
    for _ in train.starmap(train_args):
        pass

    print("\n========== Random seed sweep complete ==========")
    print("Evaluate with:")
    for (_, mask_label) in mask_results:
        print(f"  modal run modal_train_splatfacto.py::run_eval \\")
        print(f"    --lambda-depth {lambda_depth} --mask-label '{mask_label}'")


# ---------------------------------------------------------------------------
# prepare_kitti_seq — data prep for a new KITTI sequence (runs on Modal GPU)
#
# Generates:
#   DATA_MOUNT/nerfstudio/<slug>_sparse_every2/transforms.json  (from COLMAP)
#   DATA_MOUNT/kitti/.../depths_da2/                            (DA2 + GT align)
#
# Usage:
#   modal run modal_train_splatfacto.py::run_new_seq_experiments \
#     --kitti-seq-dir "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt"
# ---------------------------------------------------------------------------
@app.function(
    image=_da2_image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    volumes={DATA_MOUNT: data_vol},
)
def prepare_kitti_seq(
    kitti_seq_dir: str,
    encoder: str = "vitl",
    hold_every: int = 10,
) -> str:
    """Generate transforms.json (from COLMAP) and depths_da2 (DA2 + align) for a new sequence.

    Returns the slug (e.g. 'kitti_seq05_0018') so the caller can derive experiment names.
    """
    import os
    import re
    import subprocess
    import sys

    sys.path.insert(0, "/opt/da2")
    sys.path.insert(0, "/opt/project_scripts")

    sparse_root = f"{DATA_MOUNT}/kitti/kitti_select_static_5seq_sparse_every2"
    seq_dir = f"{sparse_root}/{kitti_seq_dir}"
    img_dir = f"{seq_dir}/images"
    gt_dir = f"{seq_dir}/depths_gt"

    m = re.search(r"KITTISeq(\d+)_.*drive_(\d+)_sync", kitti_seq_dir)
    if not m:
        raise ValueError(f"Cannot derive slug from: {kitti_seq_dir}")
    slug = f"kitti_seq{m.group(1)}_{m.group(2)}"
    nerfstudio_dst = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2"

    # ---- Step 1: generate transforms.json from COLMAP (text format) --------
    # Always regenerate — the script auto-corrects for cropped images, so we
    # must not reuse a stale transforms.json that may have wrong w/h.
    print(f"[Step 1] Generating transforms.json for {slug}...")
    subprocess.run([
        "python", "/opt/project_scripts/data_prep/colmap_to_nerfstudio_transforms.py",
        "--output-dir", seq_dir,
    ], check=True)

    # Add train/val/test splits and set up the nerfstudio directory.
    # --overwrite removes any stale destination before recreating.
    subprocess.run([
        "python", "/opt/project_scripts/data_prep/make_nerfstudio_kitti_sparse.py",
        "--src", seq_dir,
        "--dst", nerfstudio_dst,
        "--images", img_dir,
        "--stride", "1",
        "--hold-every", str(hold_every),
        "--copy-images",
        "--overwrite",
    ], check=True)
    data_vol.commit()
    print(f"  transforms.json → {nerfstudio_dst}")

    # Delete the derived _da2 nerfstudio dataset so train() rebuilds it from
    # the freshly-generated transforms.json (avoids stale w/h mismatch).
    import shutil
    da2_derived = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2_da2"
    if os.path.exists(da2_derived):
        shutil.rmtree(da2_derived)
        data_vol.commit()
        print(f"  Cleared stale {da2_derived}")

    # ---- Step 2: DA2 inference + align to GT -------------------------------
    da2_out = f"{seq_dir}/depths_da2"
    if os.path.exists(da2_out) and len(os.listdir(da2_out)) > 0:
        print(f"[Step 2] depths_da2 already exists — skipping")
    else:
        print(f"[Step 2] Running DA2 ({encoder}) inference...")
        from huggingface_hub import hf_hub_download
        ckpt_map = {
            "vits": ("depth-anything/Depth-Anything-V2-Small", "depth_anything_v2_vits.pth"),
            "vitb": ("depth-anything/Depth-Anything-V2-Base", "depth_anything_v2_vitb.pth"),
            "vitl": ("depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth"),
        }
        repo_id, filename = ckpt_map[encoder]
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir="/tmp/da2_ckpt")
        print(f"  checkpoint: {ckpt_path}")

        raw_dir = f"{seq_dir}/depths_da2_raw_npy"
        os.makedirs(raw_dir, exist_ok=True)
        subprocess.run([
            "python", "/opt/da2/run_da2_save_npy.py",
            "--img-dir", img_dir,
            "--out-dir", raw_dir,
            "--encoder", encoder,
            "--checkpoint", ckpt_path,
        ], check=True, cwd="/opt/da2")

        os.makedirs(da2_out, exist_ok=True)
        subprocess.run([
            "python", "/opt/da2/align_da2_to_kitti.py",
            "--da2-npy-dir", raw_dir,
            "--gt-depth-dir", gt_dir,
            "--out-dir", da2_out,
        ], check=True)
        data_vol.commit()
        print(f"  depths_da2 → {da2_out}")

    return slug


@app.local_entrypoint()
def run_new_seq_experiments(
    kitti_seq_dir: str = "KITTISeq05_2011_09_30_drive_0018_sync_llffdtu_s400_e725_densegt",
    lambda_depth: float = 0.1,
    ref_threshold: float = 0.18,
    encoder: str = "vitl",
    depth_loss_type: str = "mse",
    max_num_iterations: int = 50000,
):
    """Three-experiment ablation on a new KITTI sequence.

    Stage 1: prepare data (transforms.json + depths_da2) on Modal.
    Stage 2: train RGB-only (lambda=0) and Global-depth (lambda) in parallel.
    Stage 3: generate low018 mask from the Global-depth checkpoint, train masked run.

    Experiment names produced:
      <slug>_sparse_every2_da2_lambda0.0_nomask_<iters>   (RGB-only)
      <slug>_sparse_every2_da2_lambda<l>_nomask_<iters>   (Global depth)
      <slug>_sparse_every2_da2_lambda<l>_low018_<iters>   (Low-error mask)
    """
    print(f"\nPreparing new sequence: {kitti_seq_dir}")

    # Stage 1: data prep
    slug = prepare_kitti_seq.remote(kitti_seq_dir, encoder=encoder)
    print(f"  slug: {slug}")

    nomask_exp = (
        f"{slug}_sparse_every2_da2_lambda{lambda_depth}_nomask_{max_num_iterations}"
    )

    # Stage 2: RGB-only + Global depth in parallel
    print("\n[Stage 2] Training RGB-only and Global-depth in parallel...")
    for _ in train.starmap([
        (kitti_seq_dir, 0.0, depth_loss_type, "da2", max_num_iterations,
         "", "low", ref_threshold, "kitti", "bicycle", ""),
        (kitti_seq_dir, lambda_depth, depth_loss_type, "da2", max_num_iterations,
         "", "low", ref_threshold, "kitti", "bicycle", ""),
    ]):
        pass

    # Stage 3: generate low018 mask from Global-depth checkpoint, then retrain
    thresh_tag = f"{ref_threshold:.2f}".replace(".", "")
    print(f"\n[Stage 3] Generating low{thresh_tag} mask from {nomask_exp}...")
    mask_dir = generate_masks.remote(nomask_exp, ref_threshold, photo_mask_mode="low")

    print(f"\n[Stage 3] Training low{thresh_tag} masked run...")
    train.remote(
        kitti_seq_dir, lambda_depth, depth_loss_type, "da2",
        max_num_iterations, mask_dir, "low", ref_threshold,
        "kitti", "bicycle",
    )

    print("\n========== New-sequence experiments complete ==========")
    print("Evaluate with:")
    print(f"\n  # RGB-only")
    print(f"  modal run modal_train_splatfacto.py::run_eval \\")
    print(f"    --kitti-seq-dir '{kitti_seq_dir}' --lambda-depth 0.0")
    print(f"\n  # Global depth (nomask)")
    print(f"  modal run modal_train_splatfacto.py::run_eval \\")
    print(f"    --kitti-seq-dir '{kitti_seq_dir}' --lambda-depth {lambda_depth}")
    print(f"\n  # Low-error mask τ={ref_threshold}")
    print(f"  modal run modal_train_splatfacto.py::run_eval \\")
    print(f"    --kitti-seq-dir '{kitti_seq_dir}' --lambda-depth {lambda_depth} \\")
    print(f"    --photo-mask-threshold {ref_threshold} --masked")


# ---------------------------------------------------------------------------
# run_gt_comparison — rebuttal DN-Splatter comparison
#
# Runs two parallel baselines using LiDAR GT depth (depths_gt) instead of DA2:
#   1. GT + L1 loss  (closest to DN-Splatter's depth-supervision component)
#   2. GT + MSE loss (matches our existing hyperparameter for clean ablation)
#
# Prerequisites (run once before this):
#   modal volume put kitti-nerf-data \
#     data/kitti/kitti_select_static_5seq_sparse_every2/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_gt \
#     kitti/kitti_select_static_5seq_sparse_every2/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/depths_gt
#
# Usage:
#   modal run modal_train_splatfacto.py::run_gt_comparison
#   modal run modal_train_splatfacto.py::run_gt_comparison --lambda-depth 0.05
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def run_gt_comparison(
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
    lambda_depth: float = 0.1,
    max_num_iterations: int = 50000,
):
    """Train two GT-depth baselines in parallel for DN-Splatter rebuttal comparison.

    Experiment names produced:
      kitti_seq02_0034_sparse_every2_gt_lambda<l>_nomask_<iters>  (MSE)
      kitti_seq02_0034_sparse_every2_gt_lambda<l>_nomask_<iters>  (L1)
    These are differentiated by the depth_loss_type stored in config.yml.
    """
    import json

    print(f"\nDN-Splatter rebuttal comparison — GT depth baselines")
    print(f"kitti_seq_dir:   {kitti_seq_dir}")
    print(f"lambda_depth:    {lambda_depth}")
    print(f"max_iterations:  {max_num_iterations}")
    print("\nLaunching GT-MSE and GT-L1 in parallel...")

    # Tuple order matches train():
    #   kitti_seq_dir, lambda_depth, depth_loss_type, depth_sup_type,
    #   max_num_iterations, photo_mask_dir, photo_mask_mode, photo_mask_threshold,
    #   dataset_family, mip360_scene, mask_label, seed
    # mask_label differentiates the two runs in the output folder name.
    # GT-MSE gets "nomask" (default); GT-L1 gets "l1" so folders don't collide.
    configs = [
        (kitti_seq_dir, lambda_depth, "mse", "gt", max_num_iterations,
         "", "low", 0.12, "kitti", "bicycle", "nomask", 0),
        (kitti_seq_dir, lambda_depth, "l1",  "gt", max_num_iterations,
         "", "low", 0.12, "kitti", "bicycle", "l1",     0),
    ]
    list(train.starmap(configs))

    _, slug, _, _, sparse_tag = _resolve_dataset_paths(
        "kitti", kitti_seq_dir, "bicycle", "gt"
    )
    mse_exp = _experiment_name(slug, sparse_tag, "gt", lambda_depth, max_num_iterations,
                               mask_label="nomask")
    l1_exp  = _experiment_name(slug, sparse_tag, "gt", lambda_depth, max_num_iterations,
                               mask_label="l1")

    print("\n========== GT comparison runs complete ==========")
    print(f"  MSE exp: {mse_exp}")
    print(f"  L1  exp: {l1_exp}")
    print("\nEvaluate with:")
    print(f"  modal run modal_train_splatfacto.py::run_eval \\")
    print(f"    --depth-sup-type gt --lambda-depth {lambda_depth} --mask-label nomask")
    print(f"  modal run modal_train_splatfacto.py::run_eval \\")
    print(f"    --depth-sup-type gt --lambda-depth {lambda_depth} --mask-label l1")


# ---------------------------------------------------------------------------
# Render the test split of a trained experiment with ns-render dataset and
# save into compare_renders/<tag>/<exp_name>/test/rgb/.
# Use render_named_views to drive a grid of (threshold x lambda) renders.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 1,
    volumes={DATA_MOUNT: data_vol, OUT_MOUNT: out_vol},
    env={"TORCHDYNAMO_DISABLE": "1"},
)
def render_test_split(exp_name: str, compare_tag: str) -> str:
    import glob, subprocess

    pattern = f"{OUT_MOUNT}/{exp_name}/splatfacto-da2/*/config.yml"
    configs = sorted(glob.glob(pattern))
    if not configs:
        raise FileNotFoundError(f"No config.yml found: {pattern}")
    config_path = configs[-1]
    output_dir = f"{OUT_MOUNT}/compare_renders/{compare_tag}/{exp_name}"
    print(f"Rendering test split for {exp_name}")
    print(f"  config: {config_path}")
    print(f"  output: {output_dir}")

    subprocess.run(
        [
            "ns-render", "dataset",
            "--load-config", config_path,
            "--output-path", output_dir,
            "--split", "test",
            "--rendered-output-names", "rgb",
        ],
        check=True,
    )
    out_vol.commit()
    return output_dir


@app.local_entrypoint()
def render_named_views(
    view_filenames: str = "00000158.png",
    lambdas: str = "0.05 0.1 0.15",
    thresholds: str = "0.16 0.18 0.20 0.22",
    photo_mask_mode: str = "low",
    depth_sup_type: str = "da2",
    max_num_iterations: int = 50000,
    include_nomask: bool = False,
    include_masked: bool = True,
    compare_tag: str = "",
    dataset_family: str = "kitti",
    mip360_scene: str = "bicycle",
    kitti_seq_dir: str = "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
):
    """Render a specific test view across a grid of (threshold, lambda) experiments.

    Uses ns-render dataset which renders all test views into
      compare_renders/<tag>/<exp_name>/test/rgb/<view_filename>
    After download, pick the desired view filename for the paper figure.
    """
    _, slug, _, _, sparse_tag = _resolve_dataset_paths(
        dataset_family, kitti_seq_dir, mip360_scene, depth_sup_type
    )

    lambda_list = [float(v) for v in lambdas.split()]
    threshold_list = [float(v) for v in thresholds.split()]
    requested_views = view_filenames.split()
    if not compare_tag:
        view_tag = "_".join(Path(v).stem for v in requested_views)
        prefix = "kitti" if dataset_family == "kitti" else mip360_scene
        compare_tag = f"{prefix}_named_{view_tag}"

    exp_names: list[str] = []
    if include_nomask:
        exp_names.extend(
            _experiment_name(
                slug, sparse_tag, depth_sup_type, lam, max_num_iterations,
            )
            for lam in lambda_list
        )
    if include_masked:
        for thresh in threshold_list:
            exp_names.extend(
                _experiment_name(
                    slug, sparse_tag, depth_sup_type, lam, max_num_iterations,
                    photo_mask_mode=photo_mask_mode,
                    photo_mask_threshold=thresh,
                    masked=True,
                )
                for lam in lambda_list
            )

    print(f"Compare tag:     {compare_tag}")
    print(f"Requested views: {requested_views}")
    print(f"\nRendering {len(exp_names)} experiments in parallel:")
    for name in exp_names:
        print(f"  {name}")

    output_dirs = list(
        render_test_split.starmap([(name, compare_tag) for name in exp_names])
    )

    print("\n========== Render Complete ==========")
    print(f"Modal outputs:   nerf-outputs/compare_renders/{compare_tag}/")
    for name, out in zip(exp_names, output_dirs):
        print(f"  {name} -> {out}")
    print("\nDownload:")
    print(
        f"  modal volume get nerf-outputs compare_renders/{compare_tag} "
        f"./local_outputs/compare_renders/{compare_tag}"
    )
