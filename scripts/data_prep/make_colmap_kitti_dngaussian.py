#!/usr/bin/env python3
"""Convert a nerfstudio-format KITTI dataset to COLMAP format for DNGaussian.

DNGaussian reads camera poses from COLMAP sparse/ binaries (cameras.bin +
images.bin), not from poses_bounds.npy.  poses_bounds.npy is only used for
the spiral rendering path (CreateLLFFSpiral) which we don't need.

Input:
  --src           nerfstudio dataset dir (contains transforms.json + images/)
  --dst           output directory
  --depth-dir     optional dir with uint16 PNG depth maps named <stem>.png
  --overwrite     overwrite dst if it already exists

Output layout:
  <dst>/
    sparse/0/
      cameras.bin      (single PINHOLE camera)
      images.bin       (one entry per frame, no 2-D points)
      points3D.bin     (empty — DNGaussian uses random or depth init)
    images/
      <name>.png       relative symlinks → src images
      depth_maps/
        depth_<stem>.png   uint16 PNG copied from --depth-dir
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Convention helpers
# ---------------------------------------------------------------------------

# Nerfstudio uses OpenGL camera convention (x right, y up, z back).
# COLMAP uses OpenCV (x right, y down, z forward).
# Right-multiplying c2w by this matrix flips the camera's y and z axes.
_GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → COLMAP quaternion [w, x, y, z] (Shepperd's method)."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        r = np.sqrt(1.0 + t)
        s = 0.5 / r
        return np.array([0.5 * r,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        r = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        s = 0.5 / r
        return np.array([(R[2, 1] - R[1, 2]) * s,
                         0.5 * r,
                         (R[0, 1] + R[1, 0]) * s,
                         (R[0, 2] + R[2, 0]) * s])
    elif R[1, 1] > R[2, 2]:
        r = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        s = 0.5 / r
        return np.array([(R[0, 2] - R[2, 0]) * s,
                         (R[0, 1] + R[1, 0]) * s,
                         0.5 * r,
                         (R[1, 2] + R[2, 1]) * s])
    else:
        r = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        s = 0.5 / r
        return np.array([(R[1, 0] - R[0, 1]) * s,
                         (R[0, 2] + R[2, 0]) * s,
                         (R[1, 2] + R[2, 1]) * s,
                         0.5 * r])


def c2w_to_colmap(c2w_gl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert nerfstudio c2w (4×4, OpenGL) to COLMAP R (3×3) and t (3,)."""
    c2w_cv = c2w_gl @ _GL_TO_CV
    w2c = np.linalg.inv(c2w_cv)
    return w2c[:3, :3], w2c[:3, 3]


# ---------------------------------------------------------------------------
# COLMAP binary writers
# ---------------------------------------------------------------------------

def write_cameras_bin(path: Path,
                      width: int, height: int,
                      fx: float, fy: float,
                      cx: float, cy: float) -> None:
    """Write a single PINHOLE camera to cameras.bin."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))           # num_cameras
        f.write(struct.pack("<I", 1))           # camera_id = 1
        f.write(struct.pack("<i", 1))           # model_id: PINHOLE = 1
        f.write(struct.pack("<Q", width))
        f.write(struct.pack("<Q", height))
        f.write(struct.pack("<4d", fx, fy, cx, cy))


def write_images_bin(path: Path,
                     images_data: list[tuple]) -> None:
    """Write images.bin.

    images_data: list of (image_id, qvec[4], tvec[3], camera_id, name_str)
    No 2-D point observations (num_points2D = 0).
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images_data)))
        for image_id, qvec, tvec, camera_id, name in images_data:
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<4d", *qvec))  # qw, qx, qy, qz
            f.write(struct.pack("<3d", *tvec))  # tx, ty, tz
            f.write(struct.pack("<I", camera_id))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))       # num_points2D


def write_points3d_bin(path: Path) -> None:
    """Write an empty points3D.bin (DNGaussian uses depth/random init)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 0))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert nerfstudio KITTI dataset to COLMAP format for DNGaussian."
    )
    p.add_argument("--src", type=Path, required=True,
                   help="Nerfstudio dataset dir (transforms.json + images/).")
    p.add_argument("--dst", type=Path, required=True,
                   help="Output COLMAP dataset directory.")
    p.add_argument("--depth-dir", type=Path, default=None,
                   help="Dir with uint16 PNG depth maps named <stem>.png. "
                        "Copied to <dst>/images/depth_maps/depth_<stem>.png.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite the destination directory if it exists.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Source directory not found: {args.src}")
    if args.depth_dir is not None and not args.depth_dir.is_dir():
        raise SystemExit(f"Depth directory not found: {args.depth_dir}")

    # Load transforms.json
    tf_path = args.src / "transforms.json"
    if not tf_path.exists():
        raise SystemExit(f"Missing transforms.json: {tf_path}")
    with tf_path.open() as f:
        meta = json.load(f)

    h = int(meta["h"])
    w = int(meta["w"])
    fx = float(meta["fl_x"])
    fy = float(meta.get("fl_y", fx))
    cx = float(meta.get("cx", w / 2.0))
    cy = float(meta.get("cy", h / 2.0))

    frames = sorted(meta["frames"], key=lambda fr: fr["file_path"])
    if not frames:
        raise SystemExit("transforms.json contains no frames.")

    # Setup destination
    if args.dst.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Destination already exists: {args.dst}\n"
                "Use --overwrite to replace it."
            )
        if args.dst.is_symlink() or args.dst.is_file():
            args.dst.unlink()
        else:
            shutil.rmtree(args.dst)

    sparse_dir = args.dst / "sparse" / "0"
    sparse_dir.mkdir(parents=True)

    # cameras.bin — one shared PINHOLE camera
    write_cameras_bin(sparse_dir / "cameras.bin", w, h, fx, fy, cx, cy)

    # images.bin — one entry per frame
    images_data = []
    for idx, frame in enumerate(frames):
        src_img = (args.src / frame["file_path"]).resolve()
        if not src_img.exists():
            raise SystemExit(f"Source image not found: {src_img}")
        c2w_gl = np.array(frame["transform_matrix"], dtype=np.float64)
        R, t = c2w_to_colmap(c2w_gl)
        qvec = rotmat_to_qvec(R)
        images_data.append((idx + 1, qvec, t, 1, src_img.name))

    write_images_bin(sparse_dir / "images.bin", images_data)
    write_points3d_bin(sparse_dir / "points3D.bin")

    # images/ — relative symlinks to source images
    images_dst = args.dst / "images"
    images_dst.mkdir()
    stems: list[str] = []
    for frame in frames:
        src_img = (args.src / frame["file_path"]).resolve()
        dst_link = images_dst / src_img.name
        dst_link.symlink_to(os.path.relpath(src_img, images_dst))
        stems.append(src_img.stem)

    # images/depth_maps/ — copy uint16 PNG depth maps
    depth_copied = 0
    if args.depth_dir is not None:
        depth_dst = images_dst / "depth_maps"
        depth_dst.mkdir()
        missing: list[str] = []
        for stem in stems:
            src_d = args.depth_dir / f"{stem}.png"
            if not src_d.exists():
                missing.append(str(src_d))
                continue
            shutil.copy2(src_d, depth_dst / f"depth_{stem}.png")
            depth_copied += 1
        if missing:
            preview = ", ".join(missing[:5])
            suffix = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            print(
                f"  WARNING: {len(missing)} depth map(s) not found: "
                f"{preview}{suffix}"
            )

    # Summary
    print("Done.")
    print(f"  frames      : {len(frames)}")
    print(f"  image size  : {w} x {h}")
    print(f"  focal (fx,fy): {fx:.4f}, {fy:.4f}")
    print(f"  principal   : ({cx:.2f}, {cy:.2f})")
    print(f"  cameras.bin : 1 PINHOLE camera  →  {sparse_dir / 'cameras.bin'}")
    print(f"  images.bin  : {len(frames)} images  →  {sparse_dir / 'images.bin'}")
    print(f"  points3D.bin: empty  →  {sparse_dir / 'points3D.bin'}")
    print(f"  images dir  : {images_dst}")
    if args.depth_dir is not None:
        print(f"  depth maps  : {depth_copied}/{len(frames)} copied from {args.depth_dir}")
    else:
        print(f"  depth maps  : skipped (no --depth-dir provided)")


if __name__ == "__main__":
    main()
