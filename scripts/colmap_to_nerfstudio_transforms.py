#!/usr/bin/env python3
"""Build transforms.json from existing COLMAP sparse/0 (no ns-process-data / nerfstudio install).

Mip-360 bicycle uses PINHOLE in COLMAP. Requires: numpy, and repo nerfstudio colmap_parsing_utils only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "nerfstudio"))
from nerfstudio.data.utils.colmap_parsing_utils import (  # noqa: E402
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)


def parse_pinhole_camera(camera) -> dict[str, Any]:
    """PINHOLE / SIMPLE_PINHOLE intrinsics for transforms.json."""
    p = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        fl_x = fl_y = float(p[0])
        cx, cy = float(p[1]), float(p[2])
    elif camera.model == "PINHOLE":
        fl_x, fl_y = float(p[0]), float(p[1])
        cx, cy = float(p[2]), float(p[3])
    else:
        raise NotImplementedError(f"Camera model {camera.model} not supported by this script.")
    return {
        "w": int(camera.width),
        "h": int(camera.height),
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "camera_model": "OPENCV",
    }


def colmap_to_transforms(recon_dir: Path, output_dir: Path) -> int:
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    if set(cam_id_to_camera.keys()) != {1}:
        raise NotImplementedError("Multi-camera COLMAP models are not supported.")

    out = parse_pinhole_camera(cam_id_to_camera[1])
    frames = []

    for im_id, im_data in im_id_to_image.items():
        rotation = qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1

        name = Path(im_data.name).name
        frames.append(
            {
                "file_path": f"./images/{name}",
                "transform_matrix": c2w.tolist(),
                "colmap_im_id": im_id,
            }
        )

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([0, 2, 1]), :]
    applied_transform[2, :] *= -1

    out["frames"] = frames
    out["applied_transform"] = applied_transform.tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "transforms.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--colmap-subdir", type=str, default="sparse/0")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    recon_dir = output_dir / args.colmap_subdir
    if not (recon_dir / "cameras.bin").exists():
        raise FileNotFoundError(recon_dir)

    if not (output_dir / "images").exists():
        raise FileNotFoundError(f"Missing {output_dir / 'images'}")

    n = colmap_to_transforms(recon_dir, output_dir)
    print(f"Wrote {output_dir / 'transforms.json'} ({n} frames)")


if __name__ == "__main__":
    main()
