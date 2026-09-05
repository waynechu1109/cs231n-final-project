"""
Modal training script for SparseGS on KITTI sparse-every-2 data.
Camera-ready comparison against splatfacto-da2 + photometric masking (ours).

SparseGS: Xiong et al. 2023 (arXiv 2312.00206) — depth-supervised 3DGS with
Pearson correlation depth loss + unseen-viewpoint regularization (UVR).
GitHub: https://github.com/ForMyCat/SparseGS
Uses COLMAP sparse/ format + depths/<stem>.npy (float32, metres).

SDS diffusion regularization is disabled (--lambda_SDS 0) to avoid loading
a full diffusion model and the associated 24 GB VRAM cost.

Train (default: GT LiDAR depth, 30k iters):
  modal run modal_train_sparsegs.py::main

Evaluate a saved checkpoint:
  modal run modal_train_sparsegs.py::run_eval

Download results:
  modal volume get nerf-outputs sparsegs_kitti_seq02_0034_sparse_every2_gt_30000 ./local_outputs/sparsegs
"""

from __future__ import annotations

import re
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Shared volumes
# ---------------------------------------------------------------------------
data_vol = modal.Volume.from_name("kitti-nerf-data", create_if_missing=True)
out_vol = modal.Volume.from_name("nerf-outputs", create_if_missing=True)

DATA_MOUNT = "/vol/data"
OUT_MOUNT = "/vol/outputs"

SCRIPTS_LOCAL = Path(__file__).parent / "scripts"

# ---------------------------------------------------------------------------
# Image — CUDA 11.8 devel + PyTorch 2.1.2 + SparseGS submodules.
# SDS disabled: no diffusers/transformers needed.
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

_with_sparsegs = _base.run_commands(
    # Clone SparseGS with submodules (diff-gaussian-rasterization-softmax + simple-knn)
    "GIT_TERMINAL_PROMPT=0 git clone --recursive "
    "https://github.com/ForMyCat/SparseGS /opt/SparseGS",
    "pip install --no-cache-dir wheel setuptools",
    # Patch setup.py to use c++17 — torch 2.1.2 headers require it.
    # Use chr(34) to embed " inside shell single-quoted -c argument.
    "python3 -c 'import pathlib; p=pathlib.Path(\"/opt/SparseGS/submodules/diff-gaussian-rasterization-softmax/setup.py\"); c=p.read_text(); c=c.replace(chr(34)+\"nvcc\"+chr(34)+\": [\",chr(34)+\"nvcc\"+chr(34)+\": [\"+chr(34)+\"-std=c++17\"+chr(34)+\", \"); c=c.replace(chr(34)+\"cxx\"+chr(34)+\": []\",chr(34)+\"cxx\"+chr(34)+\": [\"+chr(34)+\"-std=c++17\"+chr(34)+\"]\"); p.write_text(c); print(\"diff-gaussian-rasterization-softmax setup.py patched for c++17\")'",
    # GLM was not recursively fetched — clone it into the expected path.
    "GIT_TERMINAL_PROMPT=0 git clone --depth=1 "
    "https://github.com/g-truc/glm.git "
    "/opt/SparseGS/submodules/diff-gaussian-rasterization-softmax/third_party/glm",
    # Diagnostic: compile forward.cu directly to surface the actual nvcc error.
    "(cd /opt/SparseGS/submodules/diff-gaussian-rasterization-softmax && "
    "echo '=== GLM ===' && ls third_party/glm/ 2>&1 | head -4 && "
    "echo '=== setup.py nvcc/std lines ===' && grep -n 'nvcc\\|std' setup.py | head -8 && "
    "echo '=== nvcc forward.cu ===' && "
    "/usr/local/cuda/bin/nvcc -std=c++17 "
    "-I/usr/local/lib/python3.10/site-packages/torch/include "
    "-I/usr/local/lib/python3.10/site-packages/torch/include/torch/csrc/api/include "
    "-I/usr/local/cuda/include -I/usr/local/include/python3.10 "
    "-Ithird_party/glm "
    "--expt-relaxed-constexpr "
    "-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ "
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ "
    "--compiler-options=-fPIC "
    "-gencode arch=compute_86,code=sm_86 "
    "-c cuda_rasterizer/forward.cu -o /tmp/test_fwd.o 2>&1 | head -80; "
    "echo '=== diag done ===') || true",
    # diff-gaussian-rasterization-softmax (custom fork with softmax depth rendering)
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' MAX_JOBS=4 "
    "pip install --no-build-isolation --no-cache-dir "
    "/opt/SparseGS/submodules/diff-gaussian-rasterization-softmax",
    # simple-knn: try SparseGS submodule first, fall back to public mirrors
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' "
    "pip install --no-build-isolation --no-cache-dir "
    "/opt/SparseGS/submodules/simple-knn || "
    "(GIT_TERMINAL_PROMPT=0 git clone "
    "https://gitlab.inria.fr/bkerbl/simple-knn.git /opt/simple-knn && "
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' "
    "pip install --no-build-isolation --no-cache-dir /opt/simple-knn) || "
    "(GIT_TERMINAL_PROMPT=0 git clone "
    "https://github.com/camenduru/simple-knn /opt/simple-knn-cam && "
    "TORCH_CUDA_ARCH_LIST='8.0 8.6' "
    "pip install --no-build-isolation --no-cache-dir /opt/simple-knn-cam)",
    # Python deps (no diffusers/transformers — SDS is disabled at runtime)
    "pip install --no-cache-dir 'numpy<2.0.0' plyfile tqdm lpips "
    "scikit-image imageio opencv-python scipy 'diptest==0.6.1' matplotlib icecream torchmetrics recordclass",
    # Install remaining SparseGS deps from its own requirements.txt (skip already-installed ones)
    "pip install --no-cache-dir -r /opt/SparseGS/requirements.txt 2>&1 | tail -5 || true",
)

image = _with_sparsegs.add_local_dir(SCRIPTS_LOCAL, "/opt/project_scripts")

app = modal.App("sparsegs-kitti")

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
    return f"sparsegs_{slug}_sparse_every2_{depth_type}_{iters}"


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
    """Convert KITTI nerfstudio dataset → SparseGS COLMAP+npy format, then train."""
    import os
    import subprocess

    slug = _slug(kitti_seq_dir)
    sparse_root = f"{DATA_MOUNT}/kitti/kitti_select_static_5seq_sparse_every2"
    seq_dir = f"{sparse_root}/{kitti_seq_dir}"
    nerfstudio_src = f"{DATA_MOUNT}/nerfstudio/{slug}_sparse_every2"
    exp_name = _exp_name(slug, depth_type, max_iterations)

    if depth_type == "gt":
        depth_dir = f"{seq_dir}/depths_gt"
    else:
        depth_dir = f"{DATA_MOUNT}/da2_depths/{slug}"

    sparsegs_data = f"{DATA_MOUNT}/sparsegs/{slug}_sparse_every2_{depth_type}"
    out_dir = f"{OUT_MOUNT}/{exp_name}"

    for path, label in [
        (seq_dir, "KITTI sequence dir"),
        (nerfstudio_src, "nerfstudio source dir"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}:\n  {path}")

    if depth_type != "gt" and not os.path.exists(depth_dir):
        print(f"WARNING: DA-V2 depth dir not found: {depth_dir}. Training without depth.")
        depth_dir = None

    def _needs_rebuild(d: str) -> bool:
        if not os.path.exists(f"{d}/sparse/0/cameras.bin"):
            return True
        if not os.path.exists(f"{d}/.sparsegs_v1"):
            return True
        images_dir = f"{d}/images"
        if not os.path.isdir(images_dir):
            return True
        for fname in os.listdir(images_dir):
            if fname.endswith(".png"):
                return not os.path.isfile(f"{images_dir}/{fname}")
        return True

    if _needs_rebuild(sparsegs_data):
        print(f"Building SparseGS dataset → {sparsegs_data}")
        prep_cmd = [
            "python", "/opt/project_scripts/data_prep/make_colmap_kitti_sparsegs.py",
            "--src", nerfstudio_src,
            "--dst", sparsegs_data,
            "--overwrite",
        ]
        if depth_dir and os.path.exists(depth_dir):
            prep_cmd += ["--depth-dir", depth_dir]
        subprocess.run(prep_cmd, check=True)
        data_vol.commit()
    else:
        print(f"SparseGS dataset already exists: {sparsegs_data}")

    # Clear stale output so training starts clean
    if os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
        print(f"Cleared stale output: {out_dir}")

    has_depths = os.path.isdir(f"{sparsegs_data}/depths")

    # Patch SparseGS to disable SDS (avoids loading diffusion model + 24 GB VRAM).
    # SDS fires at ~2/3 of iterations; setting lambda_SDS=0 disables the loss but
    # the diffusion model may still be imported. Patch the import away too.
    subprocess.run(
        ["python", "-c",
         "import pathlib, re\n"
         "p = pathlib.Path('/opt/SparseGS/train.py')\n"
         "# Replace guidance/sd_utils.py with a stub — avoids importing the entire\n"
         "# diffusion model stack (transformers, diffusers, ~1 GB). The stub satisfies\n"
         "# loss_utils.py's import; with lambda_diffusion=0 it is never actually called.\n"
         "stub = pathlib.Path('/opt/SparseGS/guidance/sd_utils.py')\n"
         "stub.write_text(\n"
         "    '# SDS stub: lambda_diffusion=0, diffusion model disabled\\n'\n"
         "    'class StableDiffusion:\\n'\n"
         "    '    def __init__(self, *a, **kw): pass\\n'\n"
         "    '    def train_step(self, *a, **kw): return None, 0.0\\n'\n"
         ")\n"
         "print('guidance/sd_utils.py replaced with stub')\n"
         "c = p.read_text()\n"
         "# Force lambda_diffusion default to 0\n"
         "c2 = re.sub(r'(lambda_diffusion[^,)]*default\\s*=\\s*)[0-9.]+', r'\\g<1>0.0', c)\n"
         "p.write_text(c2)\n"
         "print('train.py SDS patch applied')\n"
        ],
        check=True,
    )

    # prune_sched checkpoints: 3 milestones spread across training
    mid = max_iterations // 2
    prune_sched = [mid, mid + max_iterations // 6, mid + max_iterations // 3]

    cmd = [
        "python", "/opt/SparseGS/train.py",
        "--source_path", sparsegs_data,
        "--model_path", out_dir,
        "--eval",                          # hold out every 8th frame for test
        "--iterations", str(max_iterations),
        "--lambda_diffusion", "0",         # disable SDS diffusion regularization
        "--lambda_pearson", "0.05",
        "--lambda_local_pearson", "0.15",
        "--box_p", "128",
        "--beta", "5.0",
        "--prune_sched", *[str(s) for s in prune_sched],
    ]

    if not has_depths:
        # No depth maps: zero out Pearson weights so training still runs
        cmd += ["--lambda_pearson", "0", "--lambda_local_pearson", "0"]

    print("=" * 60)
    print(f"Exp name   : {exp_name}")
    print(f"Data dir   : {sparsegs_data}")
    print(f"Out dir    : {out_dir}")
    print(f"Depths     : {'yes' if has_depths else 'no'}")
    print(f"Iters      : {max_iterations}")
    print(f"Command:\n  " + " \\\n  ".join(cmd))
    print("=" * 60)

    subprocess.run(cmd, cwd="/opt/SparseGS", check=True)
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
    import glob
    import json
    import os
    import re
    import subprocess

    import numpy as np

    out_dir = f"{OUT_MOUNT}/{exp_name}"
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"Experiment dir not found: {out_dir}")

    # Infer source_path from exp_name slug
    m = re.search(r"sparsegs_(kitti_seq\d+_\d+)_sparse_every2_(\w+)_\d+", exp_name)
    if not m:
        raise ValueError(f"Cannot parse exp_name: {exp_name}")
    slug, depth_type = m.group(1), m.group(2)
    source_path = f"{DATA_MOUNT}/sparsegs/{slug}_sparse_every2_{depth_type}"

    metrics_json = f"{out_dir}/sparsegs_metrics.json"
    if os.path.exists(metrics_json):
        with open(metrics_json) as f:
            return json.load(f)

    # Apply guidance stub so render.py can import SparseGS modules cleanly.
    subprocess.run(
        ["python", "-c",
         "import pathlib\n"
         "p = pathlib.Path('/opt/SparseGS/guidance/sd_utils.py')\n"
         "p.write_text(\n"
         "    '# SDS stub\\n'\n"
         "    'class StableDiffusion:\\n'\n"
         "    '    def __init__(self, *a, **kw): pass\\n'\n"
         "    '    def train_step(self, *a, **kw): return None, 0.0\\n'\n"
         ")\n"
         "print('eval: guidance/sd_utils.py stub applied')\n"
        ],
        check=True,
    )

    # Determine test image names from COLMAP images.bin (same logic as 3DGS --eval:
    # sorted by name, every 8th index (0-based) is test).
    import struct

    images_bin = os.path.join(source_path, "sparse", "0", "images.bin")
    all_image_names: list[str] = []
    with open(images_bin, "rb") as f:
        n_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_images):
            f.read(64)          # image_id (i) + qvec (4d) + tvec (3d) + camera_id (i) = 8+32+24 = skip whole block
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            all_image_names.append(name_bytes.decode("utf-8"))
            n_p2d = struct.unpack("<Q", f.read(8))[0]
            f.read(n_p2d * 24)  # skip point2D entries

    sorted_names = sorted(all_image_names)
    test_names  = {nm for i, nm in enumerate(sorted_names) if i % 8 == 0}
    train_names = {nm for i, nm in enumerate(sorted_names) if i % 8 != 0}
    print(f"COLMAP: {len(sorted_names)} total, {len(test_names)} test, {len(train_names)} train")
    print(f"  test sample: {sorted(test_names)[:3]}")

    # SparseGS's render.py only outputs training-set renders (every-8th test
    # cameras are never written to disk by render.py). We render the test set
    # ourselves by loading the model directly via the SparseGS Python API.
    test_renders_dir = f"{out_dir}/test_set/renders"
    test_gt_dir      = f"{out_dir}/test_set/gt"
    existing_test = sorted(glob.glob(f"{test_renders_dir}/*.png"))
    print(f"  existing custom test renders: {len(existing_test)}")

    if len(existing_test) < len(test_names):
        # Write a self-contained rendering script to /tmp and run it.
        render_script = r"""
import sys, os, glob
import numpy as np, torch
from PIL import Image
from argparse import ArgumentParser, Namespace

sys.path.insert(0, '/opt/SparseGS')
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel

model_path    = sys.argv[1]
source_path   = sys.argv[2]
renders_out   = sys.argv[3]
gt_out        = sys.argv[4]
os.makedirs(renders_out, exist_ok=True)
os.makedirs(gt_out, exist_ok=True)

# Load the configuration that was saved at training time.
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

# Build pipe manually — PipelineParams.extract() may miss SparseGS-specific
# fields (beta, etc.) that the renderer accesses.
class _Pipe:
    compute_cov3D_python = False
    convert_SHs_python   = False
    debug                = False
    beta                 = 5.0
pp = _Pipe()
for k in ('compute_cov3D_python', 'convert_SHs_python', 'debug', 'beta'):
    if k in cfg:
        setattr(pp, k, cfg[k])

gaussians = GaussianModel(mp.sh_degree)

# Use load_iteration=None so Scene reads COLMAP (not cameras.json) and
# avoids the json_cams unbound-variable bug in SparseGS's Scene.__init__.
# We load the trained PLY separately afterwards.
scene = Scene(mp, gaussians, load_iteration=None)

import re as _re
ply_paths = sorted(
    glob.glob(os.path.join(model_path, 'point_cloud', 'iteration_*', 'point_cloud.ply')),
    key=lambda p: int(_re.search(r'iteration_(\d+)', p).group(1)),
)
if not ply_paths:
    raise RuntimeError(f"No point_cloud.ply found under {model_path}/point_cloud/")
gaussians.load_ply(ply_paths[-1])
print(f'Loaded Gaussians from {ply_paths[-1]}', flush=True)

# SparseGS's gaussian_renderer expects pipe.beta (custom param not in stock PipelineParams).
if not hasattr(pp, 'beta'):
    pp.beta = cfg.get('beta', 5.0)

bg = torch.tensor([1,1,1] if mp.white_background else [0,0,0],
                  dtype=torch.float32, device='cuda')

test_cams = scene.getTestCameras()
print(f'Rendering {len(test_cams)} test cameras ...', flush=True)

for i, cam in enumerate(test_cams):
    with torch.no_grad():
        pkg = render(cam, gaussians, pp, bg)
    img = pkg['render'].clamp(0, 1)

    name = getattr(cam, 'image_name', None) or getattr(cam, 'uid', f'{i:08d}')
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
        script_path = "/tmp/render_sparsegs_test.py"
        with open(script_path, "w") as fh:
            fh.write(render_script)

        print("Rendering test cameras via custom script ...")
        rend_proc = subprocess.run(
            ["python", script_path, out_dir, source_path,
             test_renders_dir, test_gt_dir],
            cwd="/opt/SparseGS",
            capture_output=True,
            text=True,
        )
        print("render script stdout:", rend_proc.stdout[-4000:])
        if rend_proc.returncode != 0:
            print("render script stderr:", rend_proc.stderr[-4000:])
            raise RuntimeError(f"Custom test render script exited {rend_proc.returncode}")

        out_vol.commit()

    all_render_pngs = sorted(glob.glob(f"{test_renders_dir}/*.png"))
    all_gt_pngs     = sorted(glob.glob(f"{test_gt_dir}/*.png"))
    if not all_render_pngs:
        raise RuntimeError(
            "Custom test rendering produced no PNG files. "
            "Check render script stdout above."
        )

    # Pair rendered and GT by matching filenames (both were saved with same name).
    render_map = {os.path.basename(p): p for p in all_render_pngs}
    gt_map     = {os.path.basename(p): p for p in all_gt_pngs}
    pairs = [(render_map[nm], gt_map[nm]) for nm in render_map if nm in gt_map]
    print(f"Computing metrics over {len(pairs)} test image pairs ...")

    # Install Pillow if not already present (it ships with most torch images).
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
    print("\n========== SparseGS Eval Metrics ==========")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("===========================================\n")

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
    print("\n========== SparseGS Eval Results ==========")
    print(json.dumps(metrics, indent=2))
    print("===========================================")
