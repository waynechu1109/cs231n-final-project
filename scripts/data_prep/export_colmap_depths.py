#!/usr/bin/env python3
"""Export sparse depth from an existing COLMAP model (e.g. Mip-NeRF 360 bicycle pack).

Writes uint16 PNGs (depth * 256) compatible with align_da2_to_kitti.py. Uses the scene's
sparse/0 reconstruction — no KITTI LiDAR required.

Usage:
  python scripts/export_colmap_depths.py \
    --scene-dir data/mip360_sparse/bicycle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "nerfstudio"))
from nerfstudio.data.utils.colmap_parsing_utils import (  # noqa: E402
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
)


def rasterize_sparse_depth(
    recon_dir: Path,
    min_depth: float = 0.001,
    max_depth: float = 10000.0,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 2,
) -> dict[str, np.ndarray]:
    """stem -> float32 camera-space z (valid only at triangulated keypoints)."""
    ptid_to_info = read_points3D_binary(recon_dir / "points3D.bin")
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    camera_id = next(iter(cam_id_to_camera))
    cam = cam_id_to_camera[camera_id]
    w, h = int(cam.width), int(cam.height)

    out: dict[str, np.ndarray] = {}
    for im_data in im_id_to_image.values():
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        if not pids:
            continue

        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
        rotation = qvec2rotmat(im_data.qvec)
        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
        uv = np.array(
            [im_data.xys[i] for i in range(len(im_data.xys)) if im_data.point3D_ids[i] != -1]
        )

        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < w)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < h)
        )
        z = z[idx]
        uv = uv[idx]

        depth = np.zeros((h, w), dtype=np.float32)
        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth[vv, uu] = z.astype(np.float32)

        out[Path(im_data.name).stem] = depth

    return out


def write_depth_png(depth: np.ndarray, path: Path, depth_scale: float = 256.0) -> None:
    depth_u16 = np.clip(depth * depth_scale, 0.0, 65535.0).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), depth_u16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        help="Scene root containing sparse/0/ (e.g. data/mip360_sparse/bicycle)",
    )
    parser.add_argument(
        "--colmap-dir",
        type=Path,
        help="COLMAP model dir (default: <scene-dir>/sparse/0)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output PNG folder (default: <scene-dir>/depths_colmap)",
    )
    args = parser.parse_args()

    if args.scene_dir is None and args.colmap_dir is None:
        parser.error("Provide --scene-dir or --colmap-dir")

    scene_dir = args.scene_dir.resolve() if args.scene_dir else None
    recon_dir = (
        args.colmap_dir.resolve()
        if args.colmap_dir
        else (scene_dir / "sparse" / "0")
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else (scene_dir / "depths_colmap")
    )

    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (recon_dir / name).exists():
            raise FileNotFoundError(recon_dir / name)

    depths = rasterize_sparse_depth(recon_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem, depth in tqdm(sorted(depths.items()), desc="COLMAP depth PNGs"):
        write_depth_png(depth, out_dir / f"{stem}.png")

    print(f"Wrote {len(depths)} maps to {out_dir}")


if __name__ == "__main__":
    main()
