#!/usr/bin/env python3
"""
Generate matched-ratio photometric masks from a trained splatfacto checkpoint.

These masks select exactly as many pixels as the low-error reference mask
(valid_depth AND e(u) < ref_threshold), but use different selection criteria:

  mode=high_matched:   select the k valid-depth pixels with the HIGHEST photometric error
  mode=random_matched: select k valid-depth pixels RANDOMLY (fixed seed for reproducibility)

where k = |{u : valid_depth(u) AND e(u) < ref_threshold}| computed per frame.

Usage:
  python scripts/generate_matched_ratio_masks.py \\
    --load-config outputs/<exp>/splatfacto-da2/<ts>/config.yml \\
    --output-dir /path/to/masks/high_error_matched_low018 \\
    --mode high_matched

  python scripts/generate_matched_ratio_masks.py \\
    --load-config outputs/<exp>/splatfacto-da2/<ts>/config.yml \\
    --output-dir /path/to/masks/random_matched_low018_seed0 \\
    --mode random_matched \\
    --seed 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NERFSTUDIO_ROOT = PROJECT_ROOT / "nerfstudio"
sys.path.insert(0, str(NERFSTUDIO_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from attach_nerfstudio_photo_masks import strip_photo_masks_from_dataset  # noqa: E402
from nerfstudio.engine.trainer import TrainerConfig  # noqa: E402
from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched-ratio photometric masks from a splatfacto checkpoint.")
    p.add_argument("--load-config", type=Path, required=True, help="Path to config.yml from a finished run.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write mask PNGs.")
    p.add_argument(
        "--mode",
        choices=("high_matched", "random_matched"),
        required=True,
        help=(
            "high_matched: select the k valid-depth pixels with highest photometric error. "
            "random_matched: randomly select k valid-depth pixels."
        ),
    )
    p.add_argument(
        "--ref-threshold",
        type=float,
        default=0.18,
        help="L1 RGB threshold for the reference low-error mask (default: 0.18). Determines k per frame.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used in random_matched mode (default: 0).",
    )
    p.add_argument("--load-step", type=int, default=None, help="Checkpoint step (default: latest in config).")
    p.add_argument(
        "--depth-min-valid",
        type=float,
        default=1.0 / 256.0,
        help="Minimum valid depth value in raw depth units (before dataparser_scale). Default: 1/256.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.load(args.load_config.read_text(), Loader=yaml.Loader)
    if not isinstance(config, TrainerConfig):
        raise SystemExit(f"Expected TrainerConfig in {args.load_config}")

    data_dir = Path(config.pipeline.datamanager.data)
    if strip_photo_masks_from_dataset(data_dir):
        print(f"Cleared stale photo mask paths from {data_dir}/transforms.json")

    def _set_load_step(cfg):
        if args.load_step is not None:
            cfg.load_step = args.load_step
        return cfg

    _config, pipeline, _ckpt, step = eval_setup(
        args.load_config,
        test_mode="val",
        update_config_callback=_set_load_step,
    )
    device = pipeline.device
    train_dataset = pipeline.datamanager.train_dataset
    pipeline.eval()
    pipeline.model.eval()

    dataparser_scale = float(train_dataset.metadata.get("dataparser_scale", 1.0))
    # Depth values in the loaded tensor are already scaled by dataparser_scale during dataset load.
    depth_threshold = args.depth_min_valid * dataparser_scale

    rng = np.random.RandomState(args.seed)

    total_valid_pixels = 0
    total_k_ref = 0
    total_k_new = 0
    n_frames = len(train_dataset)

    print(f"Checkpoint step:   {step}")
    print(f"Train images:      {n_frames}")
    print(f"Mode:              {args.mode}")
    print(f"ref_threshold:     {args.ref_threshold}")
    print(f"seed:              {args.seed}")
    print(f"dataparser_scale:  {dataparser_scale:.6f}")
    print(f"depth_threshold:   {depth_threshold:.6f}")
    print(f"Output dir:        {args.output_dir}")
    print()

    with torch.no_grad():
        for idx in range(n_frames):
            batch = train_dataset.get_data(idx, image_type="float32")
            camera = train_dataset.cameras[idx : idx + 1].to(device)
            outputs = pipeline.model.get_outputs_for_camera(camera)

            pred = outputs["rgb"].detach().cpu().numpy()
            gt = batch["image"].cpu().numpy()
            if pred.shape != gt.shape:
                h = min(pred.shape[0], gt.shape[0])
                w = min(pred.shape[1], gt.shape[1])
                pred = pred[:h, :w]
                gt = gt[:h, :w]

            photo_error = np.abs(pred - gt).mean(axis=-1)  # (H, W)
            H, W = photo_error.shape

            # --- valid depth mask ---
            if "depth_image" in batch:
                depth_img = batch["depth_image"].cpu().numpy()
                if depth_img.ndim == 3:
                    depth_img = depth_img[..., 0]
                depth_img = depth_img[:H, :W]
                valid_depth = depth_img > depth_threshold
            else:
                # No depth in batch; treat all pixels as valid (safety fallback).
                valid_depth = np.ones((H, W), dtype=bool)

            # --- reference count k from low-error mask ---
            # k = |valid_depth AND e(u) < ref_threshold|
            ref_mask = valid_depth & (photo_error < args.ref_threshold)
            k = int(ref_mask.sum())
            n_valid = int(valid_depth.sum())

            # Clamp in the degenerate case where ref count > valid pixels
            k = min(k, n_valid)

            # Flat indices of all valid-depth pixels
            valid_indices = np.where(valid_depth.ravel())[0]

            # --- select k pixels according to mode ---
            new_mask_flat = np.zeros(H * W, dtype=bool)

            if k > 0:
                if args.mode == "high_matched":
                    errors_at_valid = photo_error.ravel()[valid_indices]
                    # argpartition: indices of the k elements with the largest values
                    top_k_local = np.argpartition(errors_at_valid, -k)[-k:]
                    new_mask_flat[valid_indices[top_k_local]] = True
                else:  # random_matched
                    chosen = rng.choice(valid_indices, size=k, replace=False)
                    new_mask_flat[chosen] = True

            new_mask = new_mask_flat.reshape(H, W)
            k_actual = int(new_mask.sum())

            total_valid_pixels += n_valid
            total_k_ref += k
            total_k_new += k_actual

            out_name = train_dataset.image_filenames[idx].name
            Image.fromarray((new_mask.astype(np.uint8) * 255)).save(args.output_dir / out_name)

            n_pixels = H * W
            print(
                f"{idx + 1:04d}/{n_frames:04d} {out_name}  "
                f"n_valid={n_valid}  k_ref(low{str(args.ref_threshold).replace('.', '')})={k}  "
                f"k_new={k_actual}  ratio={k_actual / n_pixels:.4f}"
            )

    overall_ratio = total_k_ref / max(total_valid_pixels, 1)
    print()
    print("=== Mask generation summary ===")
    print(f"Total frames:                        {n_frames}")
    print(f"Total valid depth pixels:            {total_valid_pixels}")
    print(f"Total low{str(args.ref_threshold).replace('.', '')} reference pixels (k_ref): {total_k_ref}")
    print(f"Low-error mask ratio (k/valid):      {overall_ratio:.4f}")
    print(f"Total {args.mode} pixels:  {total_k_new}")
    print(f"Mode:                                {args.mode}  seed={args.seed}")
    print(f"Output:                              {args.output_dir}")


if __name__ == "__main__":
    main()
