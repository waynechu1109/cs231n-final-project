#!/usr/bin/env python3
"""
Generate fixed photometric masks from a trained splatfacto / splatfacto-da2 checkpoint.

  photo_error(u) = mean_c |render(u) - gt(u)|
  mode=high: mask=1 where error > threshold (apply depth loss on high-error pixels)
  mode=low:  mask=1 where error < threshold (apply depth loss on reliable pixels)
"""

from __future__ import annotations

import argparse
import contextlib
import json
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


@contextlib.contextmanager
def _strip_photo_masks_ctx(data_dir: Path):
    """Temporarily remove photo_mask_file_path entries from transforms.json.

    Needed when a previous run left partial photo_mask entries that cause
    nerfstudio's dataparser assertion (all-or-nothing requirement).
    """
    transforms_path = data_dir / "transforms.json"
    if not transforms_path.exists():
        yield
        return

    with transforms_path.open() as f:
        original_text = f.read()
    meta = json.loads(original_text)

    has_masks = any("photo_mask_file_path" in frame for frame in meta.get("frames", []))
    if not has_masks:
        yield
        return

    modified = dict(meta)
    modified["frames"] = [
        {k: v for k, v in frame.items() if k != "photo_mask_file_path"}
        for frame in meta["frames"]
    ]
    modified.pop("photo_mask_mode", None)
    with transforms_path.open("w") as f:
        json.dump(modified, f, indent=4)
        f.write("\n")
    print(f"[strip] Temporarily removed photo_mask entries from {transforms_path}")

    try:
        yield
    finally:
        with transforms_path.open("w") as f:
            f.write(original_text)
        print(f"[strip] Restored photo_mask entries in {transforms_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Photometric masks from a splatfacto checkpoint.")
    parser.add_argument(
        "--load-config",
        type=Path,
        required=True,
        help="Path to config.yml from a finished or partial splatfacto-da2 run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write mask PNGs (filenames match dataset images).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.12,
        help="L1 RGB threshold in [0,1], matching MipNeRF photo_mask_threshold.",
    )
    parser.add_argument(
        "--photo-mask-mode",
        choices=("high", "low"),
        default="high",
        help="high: mask where error > threshold. low: mask where error < threshold.",
    )
    parser.add_argument(
        "--load-step",
        type=int,
        default=None,
        help="Checkpoint step (default: latest in config load_dir).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory whose transforms.json may have stale photo_mask entries to strip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.load(args.load_config.read_text(), Loader=yaml.Loader)
    if not isinstance(config, TrainerConfig):
        raise SystemExit(f"Expected TrainerConfig in {args.load_config}")
    data_dir = Path(config.pipeline.datamanager.data)
    if strip_photo_masks_from_dataset(data_dir):
        print(f"Cleared stale photo mask paths from {data_dir}/transforms.json")

    def _set_load_step(config):
        if args.load_step is not None:
            config.load_step = args.load_step
        return config

    ctx = _strip_photo_masks_ctx(args.data_dir) if args.data_dir else contextlib.nullcontext()
    with ctx:
        _config, pipeline, _ckpt, step = eval_setup(
            args.load_config,
            test_mode="val",
            update_config_callback=_set_load_step,
        )
    device = pipeline.device
    train_dataset = pipeline.datamanager.train_dataset
    pipeline.eval()
    pipeline.model.eval()

    keep_ratios = []
    print(f"Checkpoint step {step}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Threshold: {args.threshold}  mode: {args.photo_mask_mode}")
    print(f"Writing to: {args.output_dir}")

    with torch.no_grad():
        for idx in range(len(train_dataset)):
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

            photo_error = np.abs(pred - gt).mean(axis=-1)
            if args.photo_mask_mode == "high":
                mask = photo_error > args.threshold
            else:
                mask = photo_error < args.threshold

            keep_ratio = float(mask.mean())
            keep_ratios.append(keep_ratio)

            out_name = train_dataset.image_filenames[idx].name
            out_path = args.output_dir / out_name
            Image.fromarray((mask.astype(np.uint8) * 255)).save(out_path)

            print(
                f"{idx + 1:04d}/{len(train_dataset):04d} {out_name} "
                f"keep_ratio={keep_ratio:.4f} err_mean={float(photo_error.mean()):.6f}"
            )

    print("Done.")
    print(f"mean_keep_ratio={float(np.mean(keep_ratios)):.6f}")
    print(f"median_keep_ratio={float(np.median(keep_ratios)):.6f}")


if __name__ == "__main__":
    main()
