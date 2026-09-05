"""
Modal training script for Depth-Regularized Gaussian Splatting on KITTI sparse-every-2 data.
Camera-ready comparison against splatfacto-da2 + photometric masking (ours).

Paper: Chung et al., "Depth-Regularized Optimization for 3D Gaussian Splatting in
Few-Shot Images", CVPRW 2024, arXiv 2311.13398
GitHub: https://github.com/robot0321/DepthRegularizedGS

Key differences from standard 3DGS:
 - Direct depth supervision: L1 loss between rendered depth and GT depth (--depth)
 - Canny-edge depth smoothness regularization (--usedepthReg)
 - Custom depth-accumulating rasterizer (diff-gaussian-rasterization-depth-acc)

We bypass ZoeDepth (monocular estimator) and use GT LiDAR depth directly.
We reuse the COLMAP + NPY depth data already prepared for SparseGS.

Train (GT LiDAR depth, 30k iters):
  modal run modal_train_depthreggsplatting.py::main

Evaluate:
  modal run modal_train_depthreggsplatting.py::run_eval
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Shared volumes
# ---------------------------------------------------------------------------
data_vol = modal.Volume.from_name("kitti-nerf-data", create_if_missing=True)
out_vol  = modal.Volume.from_name("nerf-outputs",    create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT  = "/vol/outputs"

SCRIPTS_LOCAL = Path(__file__).parent / "scripts"

# ---------------------------------------------------------------------------
# Image — CUDA 11.8 devel + PyTorch 2.1.2 + DepthRegGS deps.
# ---------------------------------------------------------------------------
_base = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install(
        "git", "wget", "build-essential", "ninja-build", "cmake",
        "clang", "libgl1", "ffmpeg", "libglib2.0-0", "python3.10-dev",
    )
    .pip_install(
        "torch==2.1.2+cu118",
        "torchvision==0.16.2+cu118",
        "numpy<2.0.0",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu118",
    )
)

_with_depthreggsplatting = _base.run_commands(
    # Clone DepthRegularizedGS (no recursive — we'll handle submodules manually)
    "GIT_TERMINAL_PROMPT=0 git clone "
    "https://github.com/robot0321/DepthRegularizedGS /opt/DepthRegGS",

    # ----- PyTorch3D -----
    # Try pre-built wheel from Meta CDN first (py310 + cu118 + pyt210).
    # Falls back gracefully if not available.
    "(pip install pytorch3d "
    "--extra-index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html "
    "2>&1 | tail -3 && echo 'PyTorch3D wheel OK') || "
    # Build from source on failure (targets pytorch3d v0.7.8 for PyTorch 2.1 compat)
    "(echo 'Wheel failed — building PyTorch3D from source ...' && "
    "GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch v0.7.8 "
    "https://github.com/facebookresearch/pytorch3d.git /opt/pytorch3d && "
    "CUB_HOME=/usr/local/cuda/include FORCE_CUDA=1 "
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' "
    "pip install --no-build-isolation --no-cache-dir /opt/pytorch3d 2>&1 | tail -10 && "
    "echo 'PyTorch3D source build OK')",

    # ----- Custom depth-accumulating rasterizer -----
    "GIT_TERMINAL_PROMPT=0 git clone --depth=1 "
    "https://github.com/robot0321/diff-gaussian-rasterization-depth-acc.git "
    "/opt/DepthRegGS/submodules/diff-gaussian-rasterization-depth-acc",
    # GLM headers — only clone if not already present as a submodule
    "(test -d /opt/DepthRegGS/submodules/diff-gaussian-rasterization-depth-acc/third_party/glm && "
    "echo 'GLM already present') || "
    "GIT_TERMINAL_PROMPT=0 git clone --depth=1 "
    "https://github.com/g-truc/glm.git "
    "/opt/DepthRegGS/submodules/diff-gaussian-rasterization-depth-acc/third_party/glm",
    # Patch setup.py to use c++17 (required for PyTorch 2.1.2 headers)
    "python3 -c \""
    "import pathlib; p=pathlib.Path('/opt/DepthRegGS/submodules/diff-gaussian-rasterization-depth-acc/setup.py'); "
    "c=p.read_text(); "
    "c=c.replace(chr(34)+'nvcc'+chr(34)+': [', chr(34)+'nvcc'+chr(34)+': ['+chr(34)+'-std=c++17'+chr(34)+', '); "
    "c=c.replace(chr(34)+'cxx'+chr(34)+': []', chr(34)+'cxx'+chr(34)+': ['+chr(34)+'-std=c++17'+chr(34)+']'); "
    "p.write_text(c); print('rasterizer setup.py patched for c++17')\"",
    "pip install wheel setuptools && "
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' MAX_JOBS=4 "
    "pip install --no-build-isolation --no-cache-dir "
    "/opt/DepthRegGS/submodules/diff-gaussian-rasterization-depth-acc",

    # ----- simple-knn -----
    "(GIT_TERMINAL_PROMPT=0 git clone https://gitlab.inria.fr/bkerbl/simple-knn.git /opt/simple-knn && "
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' "
    "pip install --no-build-isolation --no-cache-dir /opt/simple-knn) || "
    "(GIT_TERMINAL_PROMPT=0 git clone https://github.com/camenduru/simple-knn /opt/simple-knn-cam && "
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' "
    "pip install --no-build-isolation --no-cache-dir /opt/simple-knn-cam)",

    # ----- Python deps -----
    "pip install --no-cache-dir 'numpy<2.0.0' plyfile tqdm lpips "
    "scikit-image imageio opencv-python scipy matplotlib Pillow",

    # ----- Verify PyTorch3D import -----
    "python3 -c 'from pytorch3d.transforms.so3 import so3_exp_map; print(\"PyTorch3D OK\")' || "
    "echo 'WARNING: PyTorch3D import failed — will stub at runtime'",
)

image = _with_depthreggsplatting.add_local_dir(SCRIPTS_LOCAL, "/opt/project_scripts")

app = modal.App("depthreggsplatting-kitti")

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
    return f"depthreggsplatting_{slug}_sparse_every2_{depth_type}_{iters}"


# ---------------------------------------------------------------------------
# Patch: replace ZoeDepth block with GT NPY loading
# ---------------------------------------------------------------------------
# NOTE: _DATASET_PATCH uses single-quote triple strings internally to avoid
# conflicting with the outer r""" delimiter.
_DATASET_PATCH = r"""
import pathlib

p = pathlib.Path('/opt/DepthRegGS/scene/dataset_readers.py')
src = p.read_text()

# ------------------------------------------------------------------
# Injection: add a GT-depth loader helper after the last top-level import
# ------------------------------------------------------------------
gt_loader = (
    'def _load_gt_depth_npy(data_root, image_name):\n'
    '    # Load pre-computed GT depth (float32, metres) from depths/ dir.\n'
    '    import numpy as np, pathlib as _pl\n'
    '    stem = _pl.Path(image_name).stem\n'
    '    for suffix in [stem, image_name]:\n'
    '        candidate = _pl.Path(data_root) / "depths" / (suffix + ".npy")\n'
    '        if candidate.exists():\n'
    '            return np.load(str(candidate)).astype(np.float32)\n'
    '    return None\n'
)

if '_load_gt_depth_npy' not in src:
    last_import = max(
        (i for i, line in enumerate(src.splitlines())
         if line.startswith('import ') or line.startswith('from ')),
        default=0
    )
    lines = src.splitlines()
    lines.insert(last_import + 1, gt_loader)
    src = '\n'.join(lines)
    print('_load_gt_depth_npy injected.')

# ------------------------------------------------------------------
# Line-by-line replacement:
#   Detect "if model_zoe is None:" → insert GT loading at same indent
#   Skip all lines until "cam_info = CameraInfo" (which uses the vars)
# ------------------------------------------------------------------
if 'torch.hub.load' in src:
    lines_out = []
    src_lines = src.splitlines()
    i = 0
    patched = False
    while i < len(src_lines):
        stripped = src_lines[i].lstrip()
        if stripped.startswith('if model_zoe is None:'):
            # Determine indentation from this line
            indent = src_lines[i][: len(src_lines[i]) - len(stripped)]
            sub = indent + '    '  # one extra level
            lines_out.append(indent + '# --- GT depth loading (ZoeDepth bypassed) ---')
            # images_folder = {dataset_root}/images; go one level up for depths/
            lines_out.append(indent + '_data_root = os.path.dirname(images_folder)')
            lines_out.append(indent + 'gt_depth_np = _load_gt_depth_npy(_data_root, image_name)')
            lines_out.append(indent + 'if gt_depth_np is not None:')
            lines_out.append(sub   + 'import torch as _torch')
            lines_out.append(sub   + 'depthmap = _torch.from_numpy(gt_depth_np)')
            lines_out.append(sub   + 'depth_weight = _torch.from_numpy((gt_depth_np > 0).astype("float32"))')
            lines_out.append(sub   + 'depthloss = 0.0')
            # Skip until cam_info = CameraInfo (exclusive — that line is processed normally)
            i += 1
            while i < len(src_lines) and 'cam_info = CameraInfo' not in src_lines[i]:
                i += 1
            patched = True
            continue  # process cam_info = CameraInfo in the next iteration
        lines_out.append(src_lines[i])
        i += 1
    if patched:
        src = '\n'.join(lines_out)
        print('ZoeDepth -> GT NPY patch applied (line-by-line).')
    else:
        print('WARNING: "if model_zoe is None:" not found — dataset_readers.py may already be patched or has changed.')
else:
    print('No torch.hub.load found — possibly already patched.')

# ------------------------------------------------------------------
# Patch refineColmapWithIndex: remove the assertion that fails when
# some KITTI 3D points appear in only 1 camera.  Replace with a
# filter that aligns xyz with valid_totalptsidx before indexing.
# ------------------------------------------------------------------
# Patch refineColmapWithIndex: remove the assertion AND the subsequent
# valid3didx filtering block (xyz = xyz[valid3didx]).  read_points3D_binary
# returns xyz in file order (not sorted by ID), so boolean-masking with the
# sorted valid_totalptsidx is wrong.  For KITTI outdoor scenes we simply keep
# ALL 3D points in the PLY — extra points hurt nothing in depth-reg training.
assert_marker = 'assert len(valid_totalptsidx)==len(xyz)'
save_marker   = '### save in ply format'
if assert_marker in src:
    lines2 = src.splitlines()
    out2, i2, patched2 = [], 0, False
    while i2 < len(lines2):
        if lines2[i2].strip() == assert_marker:
            out2.append('    # KITTI fix: skip 3D-point filtering (assertion fails for outdoor scenes)')
            i2 += 1
            # Skip lines until "### save in ply format"
            while i2 < len(lines2) and save_marker not in lines2[i2]:
                i2 += 1
            patched2 = True
            continue
        out2.append(lines2[i2])
        i2 += 1
    if patched2:
        src = '\n'.join(out2)
        print('refineColmapWithIndex assertion + valid3didx block removed.')
    else:
        print('WARNING: assertion line not matched — check formatting')
else:
    print('WARNING: assertion not found in dataset_readers.py')

p.write_text(src)
print('dataset_readers.py patched OK.')
"""


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
    depth_type: str = "gt",
    max_iterations: int = 30000,
):
    """Train DepthRegularizedGS on KITTI sparse-every-2 data.

    Reuses the COLMAP + depth NPY data already prepared for SparseGS.
    Creates split_index.json (train/test split) if not present.
    """
    import json
    import os
    import struct
    import subprocess

    slug = _slug(kitti_seq_dir)
    exp_name = _exp_name(slug, depth_type, max_iterations)

    # Reuse SparseGS data dir (images + COLMAP sparse + depths/)
    sparsegs_data = f"{DATA_MOUNT}/sparsegs/{slug}_sparse_every2_{depth_type}"
    out_dir = f"{OUT_MOUNT}/{exp_name}"

    if not os.path.isdir(sparsegs_data):
        raise FileNotFoundError(
            f"SparseGS data dir not found: {sparsegs_data}\n"
            "Run modal_train_sparsegs.py::main first to create COLMAP + depth data."
        )

    # ---- Create split_index.json (every 8th image is test) ----
    split_json = f"{sparsegs_data}/split_index.json"
    # Always regenerate split_index.json: DepthRegGS needs 0-based positional
    # indices (position in name-sorted cam list), not COLMAP image IDs.
    images_bin = f"{sparsegs_data}/sparse/0/images.bin"
    with open(images_bin, "rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        img_ids_names = []
        for _ in range(n_images):
            img_id = struct.unpack("<i", f.read(4))[0]
            f.read(32 + 24)  # qvec + tvec
            cam_id = struct.unpack("<i", f.read(4))[0]  # noqa: F841
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            n_p2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_p2d * 24)
            img_ids_names.append((img_id, name))

    # Sort by name; use positional index i (not COLMAP image ID) as required by DepthRegGS
    sorted_entries = sorted(img_ids_names, key=lambda x: x[1])
    train_ids = [i for i, _ in enumerate(sorted_entries) if i % 8 != 0]
    test_ids  = [i for i, _ in enumerate(sorted_entries) if i % 8 == 0]
    split = {"train": train_ids, "test": test_ids}
    with open(split_json, "w") as f:
        json.dump(split, f, indent=4)
    data_vol.commit()
    print(f"Created split_index.json: {len(train_ids)} train, {len(test_ids)} test")

    # ---- Clear stale output ----
    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
        print(f"Cleared stale output: {out_dir}")

    # ---- Patch dataset_readers.py to bypass ZoeDepth ----
    subprocess.run(["python", "-c", _DATASET_PATCH], check=True)

    # ---- Verify PyTorch3D; stub if unavailable ----
    p3d_check = subprocess.run(
        ["python", "-c",
         "from pytorch3d.transforms.so3 import so3_exp_map; print('pytorch3d OK')"],
        capture_output=True, text=True,
    )
    if p3d_check.returncode != 0:
        print("PyTorch3D not importable — installing stub ...")
        subprocess.run(
            ["python", "-c",
             "import pathlib\n"
             "# Create a minimal pytorch3d stub that satisfies the imports\n"
             "import sys, types\n"
             "pkg = types.ModuleType('pytorch3d')\n"
             "transforms = types.ModuleType('pytorch3d.transforms')\n"
             "so3 = types.ModuleType('pytorch3d.transforms.so3')\n"
             "renderer = types.ModuleType('pytorch3d.renderer')\n"
             "cameras_mod = types.ModuleType('pytorch3d.renderer.cameras')\n"
             "\n"
             "import torch\n"
             "def so3_exp_map(x, eps=0.0001): return torch.eye(3).unsqueeze(0).expand(x.shape[0],-1,-1)\n"
             "def so3_log_map(x, eps=0.0001, cos_bound=0.0): return torch.zeros(x.shape[0], 3)\n"
             "def so3_relative_angle(R1, R2, cos_bound=0.0001, eps=0.0001): return torch.zeros(R1.shape[0])\n"
             "\n"
             "class SfMPerspectiveCameras:\n"
             "    def __init__(self, *a, **kw): pass\n"
             "    def get_world_to_view_transform(self): return None\n"
             "\n"
             "so3.so3_exp_map = so3_exp_map\n"
             "so3.so3_log_map = so3_log_map\n"
             "so3.so3_relative_angle = so3_relative_angle\n"
             "cameras_mod.SfMPerspectiveCameras = SfMPerspectiveCameras\n"
             "renderer.cameras = cameras_mod\n"
             "transforms.so3 = so3\n"
             "pkg.transforms = transforms\n"
             "pkg.renderer = renderer\n"
             "\n"
             "sys.modules['pytorch3d'] = pkg\n"
             "sys.modules['pytorch3d.transforms'] = transforms\n"
             "sys.modules['pytorch3d.transforms.so3'] = so3\n"
             "sys.modules['pytorch3d.renderer'] = renderer\n"
             "sys.modules['pytorch3d.renderer.cameras'] = cameras_mod\n"
             "\n"
             "# Write stub to a site-packages-like location so imports work\n"
             "import importlib, site\n"
             "sp = site.getsitepackages()[0]\n"
             "pytorch3d_dir = pathlib.Path(sp) / 'pytorch3d'\n"
             "pytorch3d_dir.mkdir(exist_ok=True)\n"
             "(pytorch3d_dir / '__init__.py').write_text('')\n"
             "(pytorch3d_dir / 'transforms').mkdir(exist_ok=True)\n"
             "(pytorch3d_dir / 'transforms' / '__init__.py').write_text('')\n"
             "(pytorch3d_dir / 'transforms' / 'so3.py').write_text(\n"
             "    'import torch\\n'\n"
             "    'def so3_exp_map(x, eps=0.0001): return torch.eye(3).unsqueeze(0).expand(x.shape[0],-1,-1)\\n'\n"
             "    'def so3_log_map(x, eps=0.0001, cos_bound=0.0): return torch.zeros(x.shape[0], 3)\\n'\n"
             "    'def so3_relative_angle(R1, R2, cos_bound=0.0001, eps=0.0001): return torch.zeros(R1.shape[0])\\n'\n"
             ")\n"
             "(pytorch3d_dir / 'renderer').mkdir(exist_ok=True)\n"
             "(pytorch3d_dir / 'renderer' / '__init__.py').write_text('')\n"
             "(pytorch3d_dir / 'renderer' / 'cameras.py').write_text(\n"
             "    'class SfMPerspectiveCameras:\\n'\n"
             "    '    def __init__(self, *a, **kw): pass\\n'\n"
             ")\n"
             "print('pytorch3d stub written to site-packages')\n"
            ],
            check=True,
        )

    # ---- Patch train.py: disable early stopping so full 30k iters run ----
    _train_patch = (
        "import pathlib\n"
        "p = pathlib.Path('/opt/DepthRegGS/train.py')\n"
        "src = p.read_text()\n"
        "early_stop = 'iteration > opt.min_iters and ema_depthloss_for_log > prev_depthloss'\n"
        "if early_stop in src:\n"
        "    src = src.replace(early_stop, 'False')\n"
        "    p.write_text(src)\n"
        "    print('Early stopping disabled in train.py.')\n"
        "else:\n"
        "    print('WARNING: early stopping marker not found in train.py — already patched?')\n"
    )
    subprocess.run(["python", "-c", _train_patch], check=True)

    cmd = [
        "python", "/opt/DepthRegGS/train.py",
        "-s", sparsegs_data,
        "--eval",
        "--model_path", out_dir,
        "--resolution", "1",
        "--iterations", str(max_iterations),
        "--depth",         # direct depth supervision (L1 against GT depth)
        "--usedepthReg",   # canny-edge depth smoothness regularization
        "--test_iterations", str(max_iterations),
        "--save_iterations", str(max_iterations),
    ]

    print("=" * 60)
    print(f"Exp name  : {exp_name}")
    print(f"Data dir  : {sparsegs_data}")
    print(f"Out dir   : {out_dir}")
    print(f"Iters     : {max_iterations}")
    print("Command:\n  " + " \\\n  ".join(cmd))
    print("=" * 60)

    subprocess.run(cmd, cwd="/opt/DepthRegGS", check=True)
    out_vol.commit()
    print(f"\nOutputs saved to Modal volume 'nerf-outputs': {exp_name}/")


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
    """Render test cameras and compute PSNR/SSIM/LPIPS."""
    import glob
    import json
    import os
    import re
    import struct
    import subprocess

    import numpy as np

    out_dir = f"{OUT_MOUNT}/{exp_name}"
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Experiment dir not found: {out_dir}")

    m = re.search(r"depthreggsplatting_(kitti_seq\d+_\d+)_sparse_every2_(\w+)_\d+", exp_name)
    if not m:
        raise ValueError(f"Cannot parse exp_name: {exp_name}")
    slug, depth_type = m.group(1), m.group(2)
    source_path = f"{DATA_MOUNT}/sparsegs/{slug}_sparse_every2_{depth_type}"

    metrics_json = f"{out_dir}/depthreggsplatting_metrics.json"
    if os.path.exists(metrics_json):
        with open(metrics_json) as f:
            return json.load(f)

    # Parse COLMAP to identify test images (every 8th in sorted order).
    images_bin = f"{source_path}/sparse/0/images.bin"
    all_image_names: list[str] = []
    with open(images_bin, "rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            f.read(4)       # image_id
            f.read(32 + 24) # qvec + tvec
            f.read(4)       # camera_id
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            all_image_names.append(name_bytes.decode("utf-8"))
            n_p2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_p2d * 24)

    sorted_names = sorted(all_image_names)
    test_names   = {nm for i, nm in enumerate(sorted_names) if i % 8 == 0}
    print(f"COLMAP: {len(sorted_names)} total, {len(test_names)} test")
    print(f"  test sample: {sorted(test_names)[:3]}")

    # Render test cameras using same custom script approach as SparseGS eval.
    test_renders_dir = f"{out_dir}/test_set/renders"
    test_gt_dir      = f"{out_dir}/test_set/gt"
    existing_test = sorted(glob.glob(f"{test_renders_dir}/*.png"))
    print(f"  existing test renders: {len(existing_test)}")

    if len(existing_test) < len(test_names):
        # Apply dataset_readers.py patch so Scene can be imported cleanly.
        subprocess.run(["python", "-c", _DATASET_PATCH], check=True)

        render_script = r"""
import sys, os, glob, re
import numpy as np, torch
from PIL import Image
from argparse import ArgumentParser, Namespace

sys.path.insert(0, '/opt/DepthRegGS')

# Stub pytorch3d if not installed
try:
    import pytorch3d
except ImportError:
    import types
    for mod_name in [
        'pytorch3d', 'pytorch3d.transforms', 'pytorch3d.transforms.so3',
        'pytorch3d.renderer', 'pytorch3d.renderer.cameras',
    ]:
        sys.modules.setdefault(mod_name, types.ModuleType(mod_name))
    so3 = sys.modules['pytorch3d.transforms.so3']
    so3.so3_exp_map = lambda x, eps=1e-4: torch.eye(3).unsqueeze(0).expand(x.shape[0],-1,-1)
    so3.so3_log_map = lambda x, eps=1e-4, cos_bound=0.: torch.zeros(x.shape[0], 3)
    so3.so3_relative_angle = lambda R1, R2, cos_bound=1e-4, eps=1e-4: torch.zeros(R1.shape[0])
    class SfMPerspectiveCameras:
        def __init__(self, *a, **kw): pass
    sys.modules['pytorch3d.renderer.cameras'].SfMPerspectiveCameras = SfMPerspectiveCameras

from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel

model_path  = sys.argv[1]
source_path = sys.argv[2]
renders_out = sys.argv[3]
gt_out      = sys.argv[4]
os.makedirs(renders_out, exist_ok=True)
os.makedirs(gt_out, exist_ok=True)

with open(os.path.join(model_path, 'cfg_args')) as fh:
    cfg = eval(fh.read(), {'Namespace': Namespace})
if isinstance(cfg, Namespace):
    cfg = vars(cfg)
cfg['source_path'] = source_path
cfg['model_path']  = model_path
cfg['eval']        = True
ns = Namespace(**cfg)

from arguments import ModelParams, PipelineParams
parser = ArgumentParser()
try:
    mp_cls = ModelParams(parser, sentinel=True)
except TypeError:
    mp_cls = ModelParams(parser)
mp = mp_cls.extract(ns)

class _Pipe:
    compute_cov3D_python = False
    convert_SHs_python   = False
    debug                = False
pp = _Pipe()
for k in ('compute_cov3D_python', 'convert_SHs_python', 'debug'):
    if k in cfg:
        setattr(pp, k, cfg[k])

gaussians = GaussianModel(mp.sh_degree)
scene = Scene(mp, gaussians, load_iteration=None)

ply_paths = sorted(
    glob.glob(os.path.join(model_path, 'point_cloud', 'iteration_*', 'point_cloud.ply')),
    key=lambda p: int(re.search(r'iteration_(\d+)', p).group(1)),
)
if not ply_paths:
    raise RuntimeError(f"No point_cloud.ply under {model_path}")
gaussians.load_ply(ply_paths[-1])
print(f'Loaded Gaussians from {ply_paths[-1]}', flush=True)

bg = torch.tensor([1,1,1] if mp.white_background else [0,0,0],
                  dtype=torch.float32, device='cuda')

test_cams = scene.getTestCameras()
print(f'Rendering {len(test_cams)} test cameras ...', flush=True)

for i, cam in enumerate(test_cams):
    with torch.no_grad():
        pkg = render(cam, gaussians, pp, bg)
    img = pkg['render'].clamp(0, 1)

    name = getattr(cam, 'image_name', None) or f'{i:08d}'
    if not str(name).endswith('.png'):
        name = str(name) + '.png'
    else:
        name = str(name)

    r_np = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(r_np).save(os.path.join(renders_out, name))

    gt = cam.original_image.clamp(0, 1)
    g_np = (gt.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(g_np).save(os.path.join(gt_out, name))
    print(f'  [{i+1}/{len(test_cams)}] {name}', flush=True)

print('Done.')
"""
        script_path = "/tmp/render_depthreggsplatting_test.py"
        with open(script_path, "w") as fh:
            fh.write(render_script)

        print("Rendering test cameras via custom script ...")
        rend_proc = subprocess.run(
            ["python", script_path, out_dir, source_path,
             test_renders_dir, test_gt_dir],
            cwd="/opt/DepthRegGS",
            capture_output=True,
            text=True,
        )
        print("render stdout:", rend_proc.stdout[-4000:])
        if rend_proc.returncode != 0:
            print("render stderr:", rend_proc.stderr[-4000:])
            raise RuntimeError(f"Custom test render script exited {rend_proc.returncode}")
        out_vol.commit()

    all_render_pngs = sorted(glob.glob(f"{test_renders_dir}/*.png"))
    all_gt_pngs     = sorted(glob.glob(f"{test_gt_dir}/*.png"))
    if not all_render_pngs:
        raise RuntimeError("No rendered test PNGs found after custom render script.")

    render_map = {os.path.basename(p): p for p in all_render_pngs}
    gt_map     = {os.path.basename(p): p for p in all_gt_pngs}
    pairs = [(render_map[nm], gt_map[nm]) for nm in render_map if nm in gt_map]
    print(f"Computing metrics over {len(pairs)} test image pairs ...")

    subprocess.run(
        ["pip", "install", "--quiet", "--no-cache-dir", "Pillow"],
        check=True,
    )
    from PIL import Image
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    import lpips as lpips_lib
    import torch

    loss_fn = lpips_lib.LPIPS(net="vgg").eval()
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for pred_path, gt_path in pairs:
        pred_np = np.array(Image.open(pred_path).convert("RGB"), dtype=np.float32) / 255.0
        gt_np   = np.array(Image.open(gt_path).convert("RGB"),   dtype=np.float32) / 255.0

        psnr_vals.append(peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0))
        ssim_vals.append(structural_similarity(gt_np, pred_np, data_range=1.0, channel_axis=2))

        pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        gt_t   = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            lpips_vals.append(loss_fn(pred_t, gt_t).item())

    results = {
        "PSNR":     float(np.mean(psnr_vals)),
        "SSIM":     float(np.mean(ssim_vals)),
        "LPIPS":    float(np.mean(lpips_vals)),
        "n_images": len(pairs),
    }
    print("\n========== DepthRegGS Eval Metrics ==========")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("=============================================\n")

    with open(metrics_json, "w") as f:
        json.dump(results, f, indent=2)
    out_vol.commit()
    return results


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
    print("\n========== DepthRegGS Eval Results ==========")
    print(json.dumps(metrics, indent=2))
    print("==============================================")
