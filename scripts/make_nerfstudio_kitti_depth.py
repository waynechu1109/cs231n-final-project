#!/usr/bin/env python3
"""Attach KITTI depth supervision folders to a Nerfstudio dataset transforms.json."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Nerfstudio dataset variant with per-frame depth_file_path entries. "
            "This mirrors the MipNeRF-360 depth folder layout (depths_gt, depths_da2, ...)."
        )
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source Nerfstudio dataset containing transforms.json.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output Nerfstudio dataset with depth paths added.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        required=True,
        help="Directory containing KITTI-style depth PNGs aligned with the source frames.",
    )
    parser.add_argument(
        "--depth-sup-type",
        type=str,
        default="da2",
        help="Depth supervision type suffix, matching MipNeRF Config.depth_sup_type (e.g. da2, gt).",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy the source images symlink/target instead of reusing it.",
    )
    parser.add_argument(
        "--copy-depths",
        action="store_true",
        help="Copy depth PNGs instead of creating a symlink.",
    )
    parser.add_argument(
        "--depth-crop-range",
        type=float,
        default=80.0,
        help="Ignore supervision depth above this value in meters, matching MipNeRF Config.depth_crop_range.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing destination directory.",
    )
    return parser.parse_args()


def frame_name(frame: dict) -> str:
    return Path(frame["file_path"]).name


def resolve_images_path(src: Path, meta: dict) -> Path:
    images_entry = src / "images"
    if images_entry.is_symlink():
        return images_entry.resolve()
    if images_entry.exists():
        return images_entry.resolve()

    example = src / meta["frames"][0]["file_path"]
    if example.is_symlink():
        return example.resolve().parent
    if example.exists():
        return example.parent.resolve()
    raise SystemExit(f"Could not resolve images directory under {src}")


def make_relative_to_cwd(path: Path) -> Path:
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path.resolve()


def main() -> None:
    args = parse_args()
    transforms_path = args.src / "transforms.json"
    if not transforms_path.exists():
        raise SystemExit(f"Missing source transforms.json: {transforms_path}")
    if not args.depth_dir.exists():
        raise SystemExit(f"Missing depth directory: {args.depth_dir}")
    if args.dst.exists():
        if not args.overwrite:
            raise SystemExit(f"Destination already exists: {args.dst}. Pass --overwrite to update it.")
    else:
        args.dst.mkdir(parents=True)

    with transforms_path.open() as f:
        meta = json.load(f)

    images_src = resolve_images_path(args.src, meta)
    depth_link_name = f"depths_{args.depth_sup_type}"
    depth_dst = args.dst / depth_link_name

    if args.copy_images:
        if (args.dst / "images").exists():
            if not (args.dst / "images").is_dir():
                raise SystemExit(f"Expected directory at {args.dst / 'images'}")
        else:
            shutil.copytree(images_src, args.dst / "images", symlinks=True)
    elif not (args.dst / "images").exists():
        os.symlink(images_src, args.dst / "images")

    if args.copy_depths:
        if depth_dst.exists():
            shutil.rmtree(depth_dst)
        shutil.copytree(args.depth_dir, depth_dst)
    elif not depth_dst.exists():
        os.symlink(args.depth_dir.resolve(), depth_dst)

    available_depths = {p.name for p in args.depth_dir.iterdir() if p.is_file()}
    frames = []
    missing = []
    for frame in meta["frames"]:
        name = frame_name(frame)
        out_frame = dict(frame)
        if name not in available_depths:
            missing.append(name)
            continue
        out_frame["depth_file_path"] = f"./{depth_link_name}/{name}"
        frames.append(out_frame)

    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"{len(missing)} frames are missing depth maps: {preview}")

    out_meta = dict(meta)
    out_meta["frames"] = frames
    out_meta["depth_sup_type"] = args.depth_sup_type
    out_meta["depth_unit_scale_factor"] = 1.0 / 256.0
    out_meta["depth_crop_range"] = args.depth_crop_range

    with (args.dst / "transforms.json").open("w") as f:
        json.dump(out_meta, f, indent=4)
        f.write("\n")

    manifest_path = args.dst / f"DEPTH_{args.depth_sup_type.upper()}_MANIFEST.txt"
    with manifest_path.open("w") as f:
        f.write(f"source_nerfstudio: {make_relative_to_cwd(args.src)}\n")
        f.write(f"depth_dir: {make_relative_to_cwd(args.depth_dir)}\n")
        f.write(f"depth_sup_type: {args.depth_sup_type}\n")
        f.write(f"depth_unit_scale_factor: {1.0 / 256.0}\n")
        f.write(f"depth_crop_range: {args.depth_crop_range}\n")
        f.write(f"frames: {len(frames)}\n")
        if "train_filenames" in meta:
            f.write(f"train: {len(meta.get('train_filenames', []))}\n")
            f.write(f"val: {len(meta.get('val_filenames', []))}\n")
            f.write(f"test: {len(meta.get('test_filenames', []))}\n")

    print(f"Wrote {args.dst}")
    print(f"depth_sup_type: {args.depth_sup_type}")
    print(f"frames with depth: {len(frames)}")


if __name__ == "__main__":
    main()
