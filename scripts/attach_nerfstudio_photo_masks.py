#!/usr/bin/env python3
"""Attach per-frame photometric masks to a nerfstudio transforms.json (MipNeRF-compatible layout)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Symlink a photometric mask folder into a nerfstudio dataset and add "
            "photo_mask_file_path entries to transforms.json."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Nerfstudio dataset with transforms.json (e.g. ..._sparse_every2_da2).",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Directory of mask PNGs named like frame images (same as MipNeRF fixed_photo_mask_dir).",
    )
    parser.add_argument(
        "--link-name",
        type=str,
        default="photo_masks",
        help="Subfolder name inside data-dir for the mask symlink.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing photo_mask paths in transforms.json.",
    )
    parser.add_argument(
        "--strip-only",
        action="store_true",
        help="Remove photo_mask_file_path entries and photo_masks link (no attach).",
    )
    return parser.parse_args()


def strip_photo_masks_from_dataset(data_dir: Path) -> bool:
    """Clear partial photo-mask metadata so nerfstudio can load the dataset again."""
    transforms_path = data_dir / "transforms.json"
    if not transforms_path.exists():
        return False

    with transforms_path.open() as f:
        meta = json.load(f)

    changed = False
    for frame in meta.get("frames", []):
        if "photo_mask_file_path" in frame:
            del frame["photo_mask_file_path"]
            changed = True
    if "photo_mask_mode" in meta:
        del meta["photo_mask_mode"]
        changed = True

    if changed:
        with transforms_path.open("w") as f:
            json.dump(meta, f, indent=4)
            f.write("\n")

    mask_link = data_dir / "photo_masks"
    if mask_link.is_symlink():
        mask_link.unlink()
        changed = True
    elif mask_link.exists():
        raise SystemExit(
            f"{mask_link} exists and is not a symlink; remove it manually before strip-only."
        )

    return changed


def frame_name(frame: dict) -> str:
    return Path(frame["file_path"]).name


def write_placeholder_mask(mask_dir: Path, name: str, data_dir: Path, frame: dict) -> None:
    """All-white mask so nerfstudio's dataparser gets one mask per frame (val/test are not trained)."""
    image_path = data_dir / Path(frame["file_path"])
    if not image_path.exists():
        raise SystemExit(f"Cannot size placeholder mask; missing image: {image_path}")
    with Image.open(image_path) as img:
        w, h = img.size
    out = mask_dir / name
    Image.fromarray(np.full((h, w), 255, dtype=np.uint8)).save(out)


def main() -> None:
    args = parse_args()
    transforms_path = args.data_dir / "transforms.json"
    if args.strip_only:
        if not transforms_path.exists():
            print(f"Skip strip-only: {transforms_path} does not exist yet")
            return

    if not transforms_path.exists():
        raise SystemExit(f"Missing {transforms_path}")

    if args.strip_only:
        if strip_photo_masks_from_dataset(args.data_dir):
            print(f"Stripped photo mask metadata from {transforms_path}")
        else:
            print(f"No photo mask metadata to strip in {args.data_dir}")
        return

    if args.mask_dir is None:
        raise SystemExit("--mask-dir is required unless using --strip-only")
    if not args.mask_dir.exists():
        raise SystemExit(f"Missing mask directory: {args.mask_dir}")

    with transforms_path.open() as f:
        meta = json.load(f)

    # Mask generation only renders train views; sparse every-2 keeps val/test in transforms.json.
    train_names = {Path(p).name for p in meta.get("train_filenames", [])}
    require_mask_for = train_names if train_names else None

    placeholders = 0
    for frame in meta["frames"]:
        name = frame_name(frame)
        if (args.mask_dir / name).exists():
            continue
        if require_mask_for is not None and name not in require_mask_for:
            write_placeholder_mask(args.mask_dir, name, args.data_dir, frame)
            placeholders += 1
            continue
        preview = name
        raise SystemExit(f"Train frame missing mask file: {preview}")

    available = {p.name for p in args.mask_dir.iterdir() if p.is_file()}
    mask_link = args.data_dir / args.link_name
    if mask_link.exists() and mask_link.is_symlink():
        mask_link.unlink()
    elif mask_link.exists():
        raise SystemExit(f"{mask_link} exists and is not a symlink; remove it or pick another --link-name.")
    os.symlink(args.mask_dir.resolve(), mask_link)

    missing = []
    attached = 0
    frames = []
    for frame in meta["frames"]:
        name = frame_name(frame)
        out_frame = dict(frame)
        if name not in available:
            missing.append(name)
            frames.append(out_frame)
            continue
        if "photo_mask_file_path" in out_frame and not args.overwrite:
            frames.append(out_frame)
            continue
        out_frame["photo_mask_file_path"] = f"./{args.link_name}/{name}"
        attached += 1
        frames.append(out_frame)

    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"{len(missing)} frames missing mask files, e.g. {preview}")

    meta["frames"] = frames
    meta["photo_mask_mode"] = meta.get("photo_mask_mode", "high")
    with transforms_path.open("w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")

    print(f"Linked masks: {mask_link} -> {args.mask_dir.resolve()}")
    print(
        f"Updated {transforms_path} ({len(frames)} frames, "
        f"{attached} photo_mask_file_path entries, {placeholders} val/test placeholders)"
    )


if __name__ == "__main__":
    main()
