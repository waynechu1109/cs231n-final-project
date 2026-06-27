#!/usr/bin/env python3
"""
Compute PSNR, SSIM, LPIPS for MipNeRF checkpoints from pre-rendered test_preds_50000/.

Test split: every 10th image starting from index 9 (indices 9,19,29,...,169).
Rendered:   test_preds_50000/color_000.png ... color_016.png
GT:         images/00000009.png ... 00000169.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def load_rgb(path: Path) -> np.ndarray:
    """Load image as float32 numpy array in [0, 1], shape [H, W, 3]."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """[H, W, 3] float32 → [1, 3, H, W] float32 tensor in [-1, 1] for LPIPS."""
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t * 2.0 - 1.0


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(1.0 / mse))


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(ssim(pred, gt, channel_axis=2, data_range=1.0))


def eval_checkpoint(
    ckpt_dir: Path,
    images_dir: Path,
    lpips_fn,
    step: int = 50000,
) -> dict:
    preds_dir = ckpt_dir / f"test_preds_{step}"
    if not preds_dir.exists():
        return None

    # test_indices: 9, 19, 29, ..., up to the number of renders
    render_files = sorted(preds_dir.glob("color_*.png"))
    n = len(render_files)
    if n == 0:
        return None

    test_indices = list(range(9, 9 + n * 10, 10))

    psnrs, ssims, lpipss = [], [], []

    for i, (render_path, gt_idx) in enumerate(zip(render_files, test_indices)):
        gt_path = images_dir / f"{gt_idx:08d}.png"
        if not gt_path.exists():
            print(f"  WARNING: GT not found: {gt_path}")
            continue

        pred = load_rgb(render_path)
        gt = load_rgb(gt_path)

        # Crop to same size if needed
        h = min(pred.shape[0], gt.shape[0])
        w = min(pred.shape[1], gt.shape[1])
        pred = pred[:h, :w]
        gt = gt[:h, :w]

        psnrs.append(psnr(pred, gt))
        ssims.append(compute_ssim(pred, gt))

        with torch.no_grad():
            lp = lpips_fn(to_tensor(pred), to_tensor(gt)).item()
        lpipss.append(lp)

    return {
        "n_images": len(psnrs),
        "psnr_mean": float(np.mean(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "lpips_mean": float(np.mean(lpipss)),
        "psnr_per_image": psnrs,
        "ssim_per_image": ssims,
        "lpips_per_image": lpipss,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(
            "/Users/waynechu/cs231n-final-project/data/kitti/kitti_select_static_5seq"
            "/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/logs"
        ),
        help="Directory containing checkpoints_* subdirs.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(
            "/Users/waynechu/cs231n-final-project/data/kitti/kitti_select_static_5seq"
            "/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/images"
        ),
    )
    parser.add_argument("--step", type=int, default=50000)
    parser.add_argument("--net", default="alex", choices=("alex", "vgg"),
                        help="LPIPS backbone (alex is faster).")
    args = parser.parse_args()

    print(f"Loading LPIPS ({args.net})...")
    lpips_fn = lpips.LPIPS(net=args.net)
    lpips_fn.eval()

    ckpt_dirs = sorted(d for d in args.logs_dir.iterdir()
                       if d.is_dir() and d.name.startswith("checkpoints_kitti"))

    all_results = {}
    print(f"\n{'Checkpoint':<55} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7} {'N':>4}")
    print("-" * 85)

    for ckpt_dir in ckpt_dirs:
        result = eval_checkpoint(ckpt_dir, args.images_dir, lpips_fn, step=args.step)
        if result is None:
            print(f"{ckpt_dir.name:<55} (no renders)")
            continue
        all_results[ckpt_dir.name] = result
        print(
            f"{ckpt_dir.name:<55} "
            f"{result['psnr_mean']:>7.3f} "
            f"{result['ssim_mean']:>7.4f} "
            f"{result['lpips_mean']:>7.4f} "
            f"{result['n_images']:>4}"
        )

    out_path = args.logs_dir / "metrics_summary.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
