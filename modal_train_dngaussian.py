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
        "clang",          # torch cpp_extension checks 'which clang++' at build time
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

# Layer 2: clone DNGaussian and compile CUDA extensions.
# Repo: github.com/Fictionarry/DNGaussian (CVPR 2024; project: fictionarry.github.io)
# GIT_TERMINAL_PROMPT=0 prevents git from hanging waiting for TTY credentials.
# Install CUDA submodules directly from their public GitHub URLs instead of
# relying on git submodule update (simple-knn .gitmodules points to INRIA
# gitlab which requires auth in Modal's build environment).
_with_dngaussian = _base.run_commands(
    # Clone main DNGaussian repo (no submodules needed; deps installed below)
    "GIT_TERMINAL_PROMPT=0 git clone https://github.com/Fictionarry/DNGaussian /opt/DNGaussian",
    # Ensure wheel + setuptools are present before --no-build-isolation builds.
    # Without wheel, setup.py's 'bdist_wheel' command is not registered.
    "pip install --no-cache-dir wheel setuptools",
    # diff-gaussian-rasterization: ashawkey fork adds depth rendering support.
    # --no-build-isolation: setup.py imports torch at build time; pip's isolated
    # build venv strips torch so we disable isolation to use the host env.
    "GIT_TERMINAL_PROMPT=0 TORCH_CUDA_ARCH_LIST='8.0 8.6' MAX_JOBS=4 "
    "pip install --no-build-isolation --no-cache-dir "
    "'git+https://github.com/ashawkey/diff-gaussian-rasterization.git'",
    # simple-knn: the authoritative source is the INRIA gitlab (public read);
    # graphdeco-inria/simple-knn does NOT exist as a standalone GitHub repo.
    # Try INRIA gitlab first; fall back to camenduru's public GitHub mirror.
    "(GIT_TERMINAL_PROMPT=0 git clone "
    "https://gitlab.inria.fr/bkerbl/simple-knn.git /opt/simple-knn) || "
    "(GIT_TERMINAL_PROMPT=0 git clone "
    "https://github.com/camenduru/simple-knn /opt/simple-knn)",
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' pip install --no-build-isolation --no-cache-dir /opt/simple-knn",
    # Pin numpy<2 to stay compatible with torch 2.1.2 ABI; plyfile>=1.1 pulls numpy>=2 otherwise.
    "pip install --no-cache-dir 'numpy<2.0.0' plyfile tqdm lpips scikit-image imageio opencv-python",
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

    # Build COLMAP dataset. Rebuild if images are missing/broken (old symlink-based
    # datasets from earlier runs won't work on Modal FUSE; we now copy images).
    def _needs_rebuild(colmap_data: str) -> bool:
        if not os.path.exists(f"{colmap_data}/sparse/0/cameras.bin"):
            return True
        if not os.path.exists(f"{colmap_data}/.colmap_v5"):
            return True  # stale format (per-image depth normalisation broke metric scale)
        images_dir = f"{colmap_data}/images"
        if not os.path.isdir(images_dir):
            return True
        for fname in os.listdir(images_dir):
            if fname.endswith(".png"):
                return not os.path.isfile(f"{images_dir}/{fname}")
        return True

    if _needs_rebuild(colmap_data):
        print(f"Building COLMAP dataset → {colmap_data}")
        prep_cmd = [
            "python", "/opt/project_scripts/data_prep/make_colmap_kitti_dngaussian.py",
            "--src", nerfstudio_src,
            "--dst", colmap_data,
            "--overwrite",
        ]
        if depth_dir and os.path.exists(depth_dir):
            prep_cmd += ["--depth-dir", depth_dir]
        subprocess.run(prep_cmd, check=True)
        data_vol.commit()
    else:
        print(f"COLMAP dataset already exists: {colmap_data}")

    # Generate poses_bounds.npy for DNGaussian's spiral render scene.
    # This file is required by CreateLLFFSpiral (visualization only; not used
    # for training loss or eval metrics).  Format: (N, 17) float64 array,
    # each row = flattened 3×5 LLFF pose + [near, far].
    # LLFF stored column order: [down, right, backward, t, hwf]
    # After the load-time permutation in load_llff.py these become [right, up, backward].
    poses_bounds_path = f"{colmap_data}/poses_bounds.npy"
    if not os.path.exists(poses_bounds_path):
        import json as _json
        import numpy as _np
        with open(f"{nerfstudio_src}/transforms.json") as _f:
            _meta = _json.load(_f)
        _H, _W = int(_meta["h"]), int(_meta["w"])
        _focal = float(_meta["fl_x"])
        _frames = sorted(_meta["frames"], key=lambda fr: fr["file_path"])
        _rows = []
        for _fr in _frames:
            _c2w = _np.array(_fr["transform_matrix"], dtype=_np.float64)
            # OpenGL c2w columns: [right, up, backward, t]
            _pose35 = _np.column_stack([
                -_c2w[:3, 1],            # col0 = down  = -up
                 _c2w[:3, 0],            # col1 = right
                 _c2w[:3, 2],            # col2 = backward
                 _c2w[:3, 3],            # col3 = translation
                _np.array([_H, _W, _focal]),  # col4 = hwf
            ])  # shape (3, 5)
            _rows.append(_np.concatenate([_pose35.flatten(), [0.1, 100.0]]))
        _pb = _np.array(_rows, dtype=_np.float64)
        _np.save(poses_bounds_path, _pb)
        data_vol.commit()
        print(f"Generated poses_bounds.npy: shape {_pb.shape}")

    out_dir = f"{OUT_MOUNT}/{exp_name}"
    # Remove stale output from previous (broken-depth) runs so training starts clean.
    if os.path.isdir(out_dir):
        import shutil as _shutil
        _shutil.rmtree(out_dir)
        print(f"Cleared stale output dir: {out_dir}")

    # DNGaussian expects depth maps at <source_path>/depth_maps/ (sibling of images/)
    has_depths = os.path.isdir(f"{colmap_data}/depth_maps")

    # train_llff.py handles train/test split internally (no --llffhold arg).
    # --n_sparse must be >0 to avoid UnboundLocalError in dataset_readers.py;
    # set to 77 (all training frames) so DNGaussian sees the same views as splatfacto.
    n_train = len([
        f for f in os.listdir(f"{colmap_data}/images") if f.endswith(".png")
    ]) - 11  # subtract eval frames (every 8th of 88 = 11)
    cmd = [
        "python", "/opt/DNGaussian/train_llff.py",
        "-s", colmap_data,
        "-m", out_dir,
        "--eval",
        "--n_sparse", str(n_train),
        "--iterations", str(max_iterations),
    ]

    print("=" * 60)
    print(f"Exp name   : {exp_name}")
    print(f"Data dir   : {colmap_data}")
    print(f"Out dir    : {out_dir}")
    print(f"Depths     : {'yes (' + depth_dir + ')' if has_depths else 'no'}")
    print(f"Iters      : {max_iterations}")
    print(f"Command:\n  " + " \\\n  ".join(cmd))
    print("=" * 60)

    # Multiple JIT-compiled CUDA encoders (gridencoder, shencoder, raymarching, …)
    # all use -std=c++14 in their backend.py load() calls, but PyTorch 2.1.2
    # headers require c++17.  Patch every backend.py under /opt/DNGaussian.
    _patch_backends = (
        "import os, pathlib;"
        "root = pathlib.Path('/opt/DNGaussian');"
        "files = list(root.rglob('backend.py'));"
        "count = 0;"
        "[(__import__('builtins').__setattr__('_f', open(p)) or True) and "
        " (c := _f.read()) and _f.close() or "
        " open(p, 'w').write(c.replace('-std=c++14', '-std=c++17')) and "
        " print(f'patched {p}') or count.__class__ for p in files];"
        "print(f'Done patching {len(files)} backend.py files')"
    )
    subprocess.run(
        ["python", "-c",
         "import pathlib\n"
         "root = pathlib.Path('/opt/DNGaussian')\n"
         "for p in root.rglob('backend.py'):\n"
         "    c = p.read_text()\n"
         "    if '-std=c++14' in c:\n"
         "        p.write_text(c.replace('-std=c++14', '-std=c++17'))\n"
         "        print(f'patched {p}')\n"
        ],
        check=True,
    )

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
        # render.py needs matplotlib (not in base image); install quietly
        subprocess.run(
            ["pip", "install", "--quiet", "--no-cache-dir", "matplotlib"],
            check=True,
        )
        # Also patch CUDA encoder backends (same as train)
        subprocess.run(
            ["python", "-c",
             "import pathlib\n"
             "root = pathlib.Path('/opt/DNGaussian')\n"
             "for p in root.rglob('backend.py'):\n"
             "    c = p.read_text()\n"
             "    if '-std=c++14' in c:\n"
             "        p.write_text(c.replace('-std=c++14', '-std=c++17'))\n"
             "        print(f'patched {p}')\n"
            ],
            check=True,
        )
        # Re-render test views then compute metrics
        print("Rendering test views ...")
        subprocess.run(
            ["python", "/opt/DNGaussian/render.py", "-m", out_dir, "--eval"],
            cwd="/opt/DNGaussian",
            check=True,
        )
        print("Computing metrics ...")
        import re as _re
        metrics_out = subprocess.run(
            ["python", "/opt/DNGaussian/metrics.py", "-m", out_dir],
            cwd="/opt/DNGaussian",
            check=True,
            capture_output=True,
            text=True,
        )
        print(metrics_out.stdout)
        out_vol.commit()

        # metrics.py prints results but doesn't write results.json; parse stdout.
        parsed: dict = {}
        for line in metrics_out.stdout.splitlines():
            m = _re.match(r"\s*(PSNR|SSIM|LPIPS|SSIM_sk)\s*:\s*([0-9.]+)", line)
            if m:
                parsed[m.group(1)] = float(m.group(2))
        if parsed:
            return parsed

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
):
    train.remote(
        kitti_seq_dir=kitti_seq_dir,
        depth_type=depth_type,
        max_iterations=max_iterations,
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
