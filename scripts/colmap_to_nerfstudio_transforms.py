#!/usr/bin/env python3
"""Build transforms.json from existing COLMAP sparse/0.

Supports both binary (.bin) and text (.txt) COLMAP formats.
Self-contained — does not require nerfstudio to be installed.
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Minimal COLMAP parsing (inlined from nerfstudio colmap_parsing_utils)
# ---------------------------------------------------------------------------

Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
_BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


class Image(_BaseImage):
    def qvec2rotmat(self):
        return _qvec2rotmat(self.qvec)


def _qvec2rotmat(qvec):
    return np.array([
        [1 - 2*qvec[2]**2 - 2*qvec[3]**2,
         2*qvec[1]*qvec[2] - 2*qvec[0]*qvec[3],
         2*qvec[3]*qvec[1] + 2*qvec[0]*qvec[2]],
        [2*qvec[1]*qvec[2] + 2*qvec[0]*qvec[3],
         1 - 2*qvec[1]**2 - 2*qvec[3]**2,
         2*qvec[2]*qvec[3] - 2*qvec[0]*qvec[1]],
        [2*qvec[3]*qvec[1] - 2*qvec[0]*qvec[2],
         2*qvec[2]*qvec[3] + 2*qvec[0]*qvec[1],
         1 - 2*qvec[1]**2 - 2*qvec[2]**2],
    ])


def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    return struct.unpack(endian + fmt, fid.read(num_bytes))


_CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
}


def read_cameras_text(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            cam_id = int(elems[0])
            cameras[cam_id] = Camera(
                id=cam_id, model=elems[1],
                width=int(elems[2]), height=int(elems[3]),
                params=np.array(list(map(float, elems[4:]))),
            )
    return cameras


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            props = _read_next_bytes(f, 24, "iiQQ")
            cam_id, model_id, width, height = props
            model_name, num_params = _CAMERA_MODEL_IDS[model_id]
            params = _read_next_bytes(f, 8 * num_params, "d" * num_params)
            cameras[cam_id] = Camera(
                id=cam_id, model=model_name,
                width=width, height=height,
                params=np.array(params),
            )
    return cameras


def read_images_text(path):
    images = {}
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line[0] == "#":
                continue
            elems = line.split()
            im_id = int(elems[0])
            qvec = np.array(list(map(float, elems[1:5])))
            tvec = np.array(list(map(float, elems[5:8])))
            cam_id = int(elems[8])
            name = elems[9]
            # second line: 2D keypoint observations (may be empty)
            pts = f.readline().split()
            if pts:
                xys = np.column_stack([
                    list(map(float, pts[0::3])),
                    list(map(float, pts[1::3])),
                ])
                p3d_ids = np.array(list(map(int, pts[2::3])))
            else:
                xys = np.zeros((0, 2), dtype=np.float64)
                p3d_ids = np.array([], dtype=np.int64)
            images[im_id] = Image(id=im_id, qvec=qvec, tvec=tvec,
                                  camera_id=cam_id, name=name,
                                  xys=xys, point3D_ids=p3d_ids)
    return images


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        n = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            props = _read_next_bytes(f, 64, "idddddddi")
            im_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            cam_id = props[8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            name = name.decode()
            n_pts = _read_next_bytes(f, 8, "Q")[0]
            raw = _read_next_bytes(f, 24 * n_pts, "ddq" * n_pts)
            xys = np.column_stack([raw[0::3], raw[1::3]])
            p3d_ids = np.array(raw[2::3])
            images[im_id] = Image(id=im_id, qvec=qvec, tvec=tvec,
                                  camera_id=cam_id, name=name,
                                  xys=xys, point3D_ids=p3d_ids)
    return images


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def _read_image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) by reading just the image header. Supports PNG and JPEG."""
    with open(path, "rb") as f:
        header = f.read(24)
    if header[:8] == b"\x89PNG\r\n\x1a\n":  # PNG
        w, h = struct.unpack(">II", header[16:24])
        return int(w), int(h)
    if header[:2] == b"\xff\xd8":  # JPEG
        with open(path, "rb") as f:
            f.read(2)
            while True:
                marker = f.read(2)
                if len(marker) < 2:
                    break
                if marker[1] in (0xC0, 0xC1, 0xC2):
                    f.read(3)  # length + precision
                    h, w = struct.unpack(">HH", f.read(4))
                    return int(w), int(h)
                length = struct.unpack(">H", f.read(2))[0]
                f.read(length - 2)
    raise ValueError(f"Unsupported image format: {path}")


def parse_pinhole_camera(camera: Camera) -> dict[str, Any]:
    p = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        fl_x = fl_y = float(p[0])
        cx, cy = float(p[1]), float(p[2])
    elif camera.model == "PINHOLE":
        fl_x, fl_y = float(p[0]), float(p[1])
        cx, cy = float(p[2]), float(p[3])
    else:
        raise NotImplementedError(f"Camera model {camera.model} not supported.")
    return {
        "w": int(camera.width), "h": int(camera.height),
        "fl_x": fl_x, "fl_y": fl_y,
        "cx": cx, "cy": cy,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        "camera_model": "OPENCV",
    }


def colmap_to_transforms(recon_dir: Path, output_dir: Path) -> int:
    if (recon_dir / "cameras.bin").exists():
        cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
        im_id_to_image = read_images_binary(recon_dir / "images.bin")
    else:
        cam_id_to_camera = read_cameras_text(recon_dir / "cameras.txt")
        im_id_to_image = read_images_text(recon_dir / "images.txt")

    # Use the first camera's intrinsics (all cameras share the same intrinsics in KITTI).
    first_cam = next(iter(cam_id_to_camera.values()))
    out = parse_pinhole_camera(first_cam)

    # If the actual images are cropped relative to COLMAP resolution, adjust cx/cy/w/h.
    # Assumes a centered crop (symmetric on each axis).
    images_dir = output_dir / "images"
    first_image = sorted(images_dir.iterdir())[0]
    import struct as _struct  # noqa: F811 (already imported at top, harmless re-import)
    # Use a minimal PNG/JPEG header read to avoid needing Pillow.
    actual_w, actual_h = _read_image_size(first_image)
    colmap_w, colmap_h = out["w"], out["h"]
    if (actual_w, actual_h) != (colmap_w, colmap_h):
        crop_x = (colmap_w - actual_w) / 2.0
        crop_y = (colmap_h - actual_h) / 2.0
        out["w"] = actual_w
        out["h"] = actual_h
        out["cx"] = out["cx"] - crop_x
        out["cy"] = out["cy"] - crop_y
        print(f"  Detected crop: COLMAP {colmap_w}×{colmap_h} → actual {actual_w}×{actual_h}")
        print(f"  Adjusted cx={out['cx']:.4f}  cy={out['cy']:.4f}")
    frames = []

    for im_data in im_id_to_image.values():
        rotation = _qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1

        name = Path(im_data.name).name
        frames.append({
            "file_path": f"./images/{name}",
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_data.id,
        })

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
    if not (recon_dir / "cameras.bin").exists() and not (recon_dir / "cameras.txt").exists():
        raise FileNotFoundError(f"No cameras.bin or cameras.txt in: {recon_dir}")

    if not (output_dir / "images").exists():
        raise FileNotFoundError(f"Missing images/: {output_dir / 'images'}")

    n = colmap_to_transforms(recon_dir, output_dir)
    print(f"Wrote {output_dir / 'transforms.json'} ({n} frames)")


if __name__ == "__main__":
    main()
