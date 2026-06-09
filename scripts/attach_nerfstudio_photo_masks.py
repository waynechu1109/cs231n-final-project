#!/usr/bin/env python3
"""Attach per-frame photometric masks to a nerfstudio transforms.json (MipNeRF-compatible layout)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


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
        required=True,
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
    return parser.parse_args()


def frame_name(frame: dict) -> str:
    return Path(frame["file_path"]).name


def main() -> None:
    args = parse_args()
    transforms_path = args.data_dir / "transforms.json"
    if not transforms_path.exists():
        raise SystemExit(f"Missing {transforms_path}")
    if not args.mask_dir.exists():
        raise SystemExit(f"Missing mask directory: {args.mask_dir}")

    available = {p.name for p in args.mask_dir.iterdir() if p.is_file()}
    mask_link = args.data_dir / args.link_name
    if mask_link.exists() and mask_link.is_symlink():
        mask_link.unlink()
    elif mask_link.exists():
        raise SystemExit(f"{mask_link} exists and is not a symlink; remove it or pick another --link-name.")
    os.symlink(args.mask_dir.resolve(), mask_link)

    with transforms_path.open() as f:
        meta = json.load(f)

    missing = []
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
        frames.append(out_frame)

    if missing:
        preview = ", ".join(missing[:5])
        print(f"WARNING: {len(missing)} frames missing mask files (likely val split), e.g. {preview}")
        if available:
            placeholder_src = args.mask_dir / sorted(available)[0]
            for name in missing:
                shutil.copy(placeholder_src, args.mask_dir / name)
            print(f"Created {len(missing)} placeholder masks copied from {placeholder_src.name}")
            available = {p.name for p in args.mask_dir.iterdir() if p.is_file()}
            frames = []
            for frame in meta["frames"]:
                name = frame_name(frame)
                out_frame = dict(frame)
                if "photo_mask_file_path" in out_frame and not args.overwrite:
                    frames.append(out_frame)
                    continue
                out_frame["photo_mask_file_path"] = f"./{args.link_name}/{name}"
                frames.append(out_frame)
        else:
            raise SystemExit(f"{len(missing)} frames missing mask files and no available masks to copy, e.g. {preview}")

    meta["frames"] = frames
    meta["photo_mask_mode"] = meta.get("photo_mask_mode", "high")
    with transforms_path.open("w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")

    print(f"Linked masks: {mask_link} -> {args.mask_dir.resolve()}")
    print(f"Updated {transforms_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
