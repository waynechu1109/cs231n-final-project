#!/usr/bin/env python3
"""Convert a nerfstudio-format KITTI dataset to LLFF format for DNGaussian.

Input:
  --src           nerfstudio dataset dir (contains transforms.json + images/)
  --dst           output directory
  --depth-dir     optional dir with uint16 PNG depth maps named like the images
  --overwrite     overwrite dst if it already exists

Output layout:
  <dst>/
    poses_bounds.npy      (N, 17) float32
    images/
      00000000.png        relative symlink -> src images
      ...
      depth_maps/
        depth_00000000.png  (uint16 PNG copied from --depth-dir)
        ...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


NEAR = 2.0
FAR = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert nerfstudio KITTI dataset to LLFF format for DNGaussian."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source nerfstudio dataset directory containing transforms.json and images/.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output directory for the LLFF dataset.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing uint16 PNG depth maps named like the images "
            "(e.g. 00000000.png). If provided, they are copied to "
            "<dst>/images/depth_maps/depth_<stem>.png."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination directory if it already exists.",
    )
    return parser.parse_args()


def load_transforms(src: Path) -> dict:
    transforms_path = src / "transforms.json"
    if not transforms_path.exists():
        raise SystemExit(f"Missing transforms.json: {transforms_path}")
    with transforms_path.open() as f:
        return json.load(f)


def build_poses_bounds(meta: dict) -> np.ndarray:
    """Build the (N, 17) poses_bounds array from nerfstudio transforms.json metadata.

    Row layout (LLFF convention):
        [h, w, f,  r00, r01, r02, t0,  r10, r11, r12, t1,  r20, r21, r22, t2,  near, far]

    The transform_matrix in nerfstudio is already OpenGL/LLFF convention
    (x right, y up, z backward), so we use the top-left 3x4 directly.
    """
    h = float(meta["h"])
    w = float(meta["w"])
    f = float(meta["fl_x"])

    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])

    rows = []
    for frame in frames:
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)  # (4, 4)
        r = c2w[:3, :3]  # (3, 3)
        t = c2w[:3, 3]   # (3,)

        # LLFF row: [h, w, f, r00, r01, r02, t0, r10, r11, r12, t1, r20, r21, r22, t2, near, far]
        row = np.array([
            h, w, f,
            r[0, 0], r[0, 1], r[0, 2], t[0],
            r[1, 0], r[1, 1], r[1, 2], t[1],
            r[2, 0], r[2, 1], r[2, 2], t[2],
            NEAR, FAR,
        ], dtype=np.float32)
        rows.append(row)

    return np.stack(rows, axis=0)  # (N, 17)


def setup_dst(dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise SystemExit(
                f"Destination already exists: {dst}\n"
                "Use --overwrite to replace it."
            )
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.mkdir(parents=True)


def link_images(src: Path, meta: dict, images_dst: Path) -> list[str]:
    """Create relative symlinks in images_dst pointing to source images.

    Returns the list of image stems in sorted order.
    """
    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])
    stems = []
    for frame in frames:
        src_image = (src / frame["file_path"]).resolve()
        if not src_image.exists():
            raise SystemExit(f"Source image not found: {src_image}")
        stem = src_image.stem
        dst_link = images_dst / src_image.name
        # Relative symlink: from images_dst to src_image
        rel_target = os.path.relpath(src_image, images_dst)
        dst_link.symlink_to(rel_target)
        stems.append(stem)
    return stems


def copy_depth_maps(depth_dir: Path, stems: list[str], depth_maps_dst: Path) -> int:
    """Copy uint16 PNG depth maps from depth_dir to depth_maps_dst.

    Naming convention: depth_<stem>.png

    Returns the number of depth maps copied.
    """
    depth_maps_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = []
    for stem in stems:
        src_depth = depth_dir / f"{stem}.png"
        if not src_depth.exists():
            missing.append(str(src_depth))
            continue
        dst_depth = depth_maps_dst / f"depth_{stem}.png"
        shutil.copy2(src_depth, dst_depth)
        copied += 1
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        print(
            f"  WARNING: {len(missing)} depth map(s) not found in --depth-dir: "
            f"{preview}{suffix}"
        )
    return copied


def main() -> None:
    args = parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Source directory not found: {args.src}")
    if args.depth_dir is not None and not args.depth_dir.is_dir():
        raise SystemExit(f"Depth directory not found: {args.depth_dir}")

    meta = load_transforms(args.src)

    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])
    n_frames = len(frames)
    if n_frames == 0:
        raise SystemExit("transforms.json contains no frames.")

    setup_dst(args.dst, args.overwrite)

    # Build and save poses_bounds.npy
    poses_bounds = build_poses_bounds(meta)
    npy_path = args.dst / "poses_bounds.npy"
    np.save(npy_path, poses_bounds)

    # Create images/ directory and symlinks
    images_dst = args.dst / "images"
    images_dst.mkdir(parents=True)
    stems = link_images(args.src, meta, images_dst)

    # Optionally copy depth maps
    depth_copied = 0
    if args.depth_dir is not None:
        depth_maps_dst = images_dst / "depth_maps"
        depth_copied = copy_depth_maps(args.depth_dir, stems, depth_maps_dst)

    # Summary
    h = int(meta["h"])
    w = int(meta["w"])
    fl_x = float(meta["fl_x"])
    print(f"Done.")
    print(f"  frames      : {n_frames}")
    print(f"  image size  : {w} x {h}")
    print(f"  focal length: {fl_x:.4f} px")
    print(f"  near / far  : {NEAR} / {FAR} m")
    print(f"  poses_bounds: {npy_path}  shape={poses_bounds.shape}")
    print(f"  images dir  : {images_dst}")
    if args.depth_dir is not None:
        print(f"  depth maps  : {depth_copied}/{n_frames} copied from {args.depth_dir}")
    else:
        print(f"  depth maps  : skipped (no --depth-dir provided)")


if __name__ == "__main__":
    main()
