#!/usr/bin/env python3
"""Convert a nerfstudio-format KITTI dataset to COLMAP + npy-depth format for SparseGS.

SparseGS (Xiong et al. 2023, arXiv 2312.00206) reads:
  - Camera poses from COLMAP sparse/ binaries (cameras.bin + images.bin + points3D.bin)
  - Depth maps from <source_path>/depths/<stem>.npy  (float32, metres, 0 = invalid)

The Pearson correlation depth loss is scale-invariant, so both GT LiDAR depths
(in metric metres) and monocular DA-V2 depths (arbitrary scale) work directly.

Input:
  --src           nerfstudio dataset dir (contains transforms.json + images/)
  --dst           output directory
  --depth-dir     optional dir with depth files:
                    • uint16 PNG  (<stem>.png) — KITTI GT LiDAR format
                    • float32 npy (<stem>.npy) — DA-V2 or pre-processed depths
  --overwrite     overwrite dst if it already exists

Output layout:
  <dst>/
    sparse/0/
      cameras.bin      (single PINHOLE camera)
      images.bin       (one entry per frame)
      points3D.bin     (back-projected from depth maps, or empty)
    images/
      <name>.png       copied from src
    depths/
      <stem>.npy       float32 depth in metres (0 = invalid)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Convention helpers
# ---------------------------------------------------------------------------

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

def write_cameras_bin(path: Path, width: int, height: int,
                      fx: float, fy: float, cx: float, cy: float) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<i", 1))   # PINHOLE = 1
        f.write(struct.pack("<Q", width))
        f.write(struct.pack("<Q", height))
        f.write(struct.pack("<4d", fx, fy, cx, cy))


def write_images_bin(path: Path, images_data: list[tuple]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images_data)))
        for image_id, qvec, tvec, camera_id, name in images_data:
            f.write(struct.pack("<I", image_id))
            f.write(struct.pack("<4d", *qvec))
            f.write(struct.pack("<3d", *tvec))
            f.write(struct.pack("<I", camera_id))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))   # num_points2D = 0


def write_points3d_bin(path: Path, points: list[tuple] | None = None) -> None:
    pts = points or []
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(pts)))
        for i, (x, y, z, r, g, b) in enumerate(pts):
            f.write(struct.pack("<Q", i + 1))
            f.write(struct.pack("<3d", x, y, z))
            f.write(struct.pack("<3B", int(r), int(g), int(b)))
            f.write(struct.pack("<d", 0.0))
            f.write(struct.pack("<Q", 0))


def backproject_depth(
    depth_m: np.ndarray,
    image_path: Path,
    c2w_gl: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    n_sample: int = 200,
) -> list[tuple]:
    """Back-project a float32 depth map (metres) to world-frame 3D points."""
    img = np.array(Image.open(image_path))
    h, w = depth_m.shape
    valid_mask = depth_m > 0
    ys, xs = np.where(valid_mask)
    if len(ys) == 0:
        return []
    idx = np.linspace(0, len(ys) - 1, min(n_sample, len(ys)), dtype=int)
    ys_s, xs_s = ys[idx], xs[idx]
    ds_s = depth_m[ys_s, xs_s]
    X_cam = (xs_s - cx) / fx * ds_s
    Y_cam = (ys_s - cy) / fy * ds_s
    Z_cam = ds_s
    P_cam_cv = np.stack([X_cam, Y_cam, Z_cam, np.ones_like(Z_cam)], axis=1)
    c2w_cv = c2w_gl @ _GL_TO_CV
    P_world = (c2w_cv @ P_cam_cv.T).T[:, :3]
    ys_c = np.clip(ys_s, 0, h - 1)
    xs_c = np.clip(xs_s, 0, w - 1)
    if img.ndim == 3:
        rgbs = img[ys_c, xs_c, :3]
    else:
        rgbs = np.stack([img[ys_c, xs_c]] * 3, axis=1)
    return [
        (float(P_world[i, 0]), float(P_world[i, 1]), float(P_world[i, 2]),
         int(rgbs[i, 0]), int(rgbs[i, 1]), int(rgbs[i, 2]))
        for i in range(len(P_world))
    ]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert nerfstudio KITTI dataset to COLMAP + npy-depth format for SparseGS."
    )
    p.add_argument("--src", type=Path, required=True,
                   help="Nerfstudio dataset dir (transforms.json + images/).")
    p.add_argument("--dst", type=Path, required=True,
                   help="Output SparseGS dataset directory.")
    p.add_argument("--depth-dir", type=Path, default=None,
                   help="Dir with depth maps: uint16 PNG (<stem>.png, KITTI GT) "
                        "or float32 npy (<stem>.npy, DA-V2). "
                        "Saved to <dst>/depths/<stem>.npy.")
    p.add_argument("--overwrite", action="store_true")
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

    if args.dst.exists():
        if not args.overwrite:
            raise SystemExit(f"Destination exists: {args.dst}\nUse --overwrite.")
        if args.dst.is_symlink() or args.dst.is_file():
            args.dst.unlink()
        else:
            shutil.rmtree(args.dst)

    sparse_dir = args.dst / "sparse" / "0"
    sparse_dir.mkdir(parents=True)

    write_cameras_bin(sparse_dir / "cameras.bin", w, h, fx, fy, cx, cy)

    images_data = []
    frame_meta: list[tuple] = []
    for idx, frame in enumerate(frames):
        src_img = (args.src / frame["file_path"]).resolve()
        if not src_img.exists():
            raise SystemExit(f"Source image not found: {src_img}")
        c2w_gl = np.array(frame["transform_matrix"], dtype=np.float64)
        R, t = c2w_to_colmap(c2w_gl)
        qvec = rotmat_to_qvec(R)
        images_data.append((idx + 1, qvec, t, 1, src_img.name))
        frame_meta.append((src_img, c2w_gl))

    write_images_bin(sparse_dir / "images.bin", images_data)
    write_points3d_bin(sparse_dir / "points3D.bin")

    # images/ — copy source images
    images_dst = args.dst / "images"
    images_dst.mkdir()
    stems: list[str] = []
    for frame in frames:
        src_img = (args.src / frame["file_path"]).resolve()
        shutil.copy2(src_img, images_dst / src_img.name)
        stems.append(src_img.stem)

    # depths/ — SparseGS expects float32 .npy files (depth in metres, 0=invalid).
    # Pearson loss is scale-invariant so both GT LiDAR and monocular depths work.
    depth_copied = 0
    all_points: list[tuple] = []
    if args.depth_dir is not None:
        depth_dst = args.dst / "depths"
        depth_dst.mkdir()
        missing: list[str] = []
        for (src_img, c2w_gl), stem in zip(frame_meta, stems):
            # Accept either uint16 PNG (KITTI GT) or float32 npy (DA-V2)
            src_png = args.depth_dir / f"{stem}.png"
            src_npy = args.depth_dir / f"{stem}.npy"
            if src_npy.exists():
                depth_m = np.load(src_npy).astype(np.float32)
            elif src_png.exists():
                raw = np.array(Image.open(src_png), dtype=np.float32)
                depth_m = raw / 256.0   # KITTI uint16 → metres
            else:
                missing.append(stem)
                continue
            np.save(depth_dst / f"{stem}.npy", depth_m)
            depth_copied += 1
            pts = backproject_depth(depth_m, src_img, c2w_gl, fx, fy, cx, cy)
            all_points.extend(pts)

        if missing:
            preview = ", ".join(missing[:5])
            suffix = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
            print(f"  WARNING: {len(missing)} depth map(s) not found: {preview}{suffix}")
        if all_points:
            write_points3d_bin(sparse_dir / "points3D.bin", all_points)

    print("Done.")
    print(f"  frames      : {len(frames)}")
    print(f"  image size  : {w} x {h}")
    print(f"  focal (fx,fy): {fx:.4f}, {fy:.4f}")
    print(f"  cameras.bin : 1 PINHOLE  →  {sparse_dir / 'cameras.bin'}")
    print(f"  images.bin  : {len(frames)} images  →  {sparse_dir / 'images.bin'}")
    print(f"  points3D.bin: {len(all_points)} points  →  {sparse_dir / 'points3D.bin'}")
    print(f"  images dir  : {images_dst}")
    if args.depth_dir is not None:
        print(f"  depths      : {depth_copied}/{len(frames)} saved to {depth_dst}")
    else:
        print(f"  depths      : skipped (no --depth-dir)")

    (args.dst / ".sparsegs_v1").touch()


if __name__ == "__main__":
    main()
