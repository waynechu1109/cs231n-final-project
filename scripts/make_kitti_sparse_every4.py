#!/usr/bin/env python3
"""Create a sparse KITTI subset by keeping every Nth frame per sequence."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


FRAME_EXTS = {".png", ".jpg", ".jpeg", ".npy", ".npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a KITTI sparse subset while preserving COLMAP text metadata."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/kitti/kitti_select_static_5seq"),
        help="Dense KITTI dataset root.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/kitti/kitti_select_static_5seq_sparse_every4"),
        help="Output sparse dataset root.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Keep frames whose zero-based frame index is divisible by this value.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of hard-linking frame assets.",
    )
    return parser.parse_args()


def is_frame_file(path: Path) -> bool:
    return path.suffix.lower() in FRAME_EXTS and path.stem.isdigit()


def keep_frame(path: Path, stride: int) -> bool:
    return is_frame_file(path) and int(path.stem) % stride == 0


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def filter_colmap_images(src: Path, dst: Path, stride: int) -> tuple[set[str], set[str]]:
    kept_names: set[str] = set()
    kept_camera_ids: set[str] = set()
    lines = src.read_text().splitlines()
    out: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            i += 1
            continue

        parts = line.split()
        if len(parts) < 10:
            out.append(line)
            i += 1
            continue

        image_name = parts[-1]
        image_path = Path(image_name)
        points_line = lines[i + 1] if i + 1 < len(lines) else ""
        if keep_frame(image_path, stride):
            out.append(line)
            out.append(points_line)
            kept_names.add(image_path.name)
            kept_camera_ids.add(parts[-2])
        i += 2

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out).rstrip() + "\n")
    return kept_names, kept_camera_ids


def filter_colmap_cameras(src: Path, dst: Path, camera_ids: set[str]) -> None:
    out: list[str] = []
    for line in src.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue

        parts = line.split()
        if parts and parts[0] in camera_ids:
            out.append(line)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out).rstrip() + "\n")


def copy_sequence(seq_src: Path, seq_dst: Path, stride: int, copy: bool) -> dict[str, int]:
    seq_dst.mkdir(parents=True, exist_ok=True)
    selected: set[str] = set()
    counts: dict[str, int] = {}

    for item in sorted(seq_src.iterdir()):
        if item.is_file() and item.name != ".DS_Store" and not item.name.startswith("._"):
            shutil.copy2(item, seq_dst / item.name)

    sparse_src = seq_src / "sparse" / "0"
    sparse_dst = seq_dst / "sparse" / "0"
    if sparse_src.exists():
        names, camera_ids = filter_colmap_images(
            sparse_src / "images.txt", sparse_dst / "images.txt", stride
        )
        filter_colmap_cameras(sparse_src / "cameras.txt", sparse_dst / "cameras.txt", camera_ids)
        points_src = sparse_src / "points3D.txt"
        if points_src.exists():
            shutil.copy2(points_src, sparse_dst / "points3D.txt")
        selected = names
        counts["sparse_images"] = len(names)
        counts["sparse_cameras"] = len(camera_ids)

    for subdir in sorted(p for p in seq_src.iterdir() if p.is_dir() and p.name != "sparse"):
        kept = 0
        for frame in sorted(subdir.iterdir()):
            if frame.is_file() and keep_frame(frame, stride):
                if not selected or frame.name in selected:
                    link_or_copy(frame, seq_dst / subdir.name / frame.name, copy)
                    kept += 1
        if kept:
            counts[subdir.name] = kept

    return counts


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if not args.src.exists():
        raise SystemExit(f"Source does not exist: {args.src}")
    if args.dst.exists():
        raise SystemExit(f"Destination already exists: {args.dst}")

    args.dst.mkdir(parents=True)
    summaries = []
    for seq_src in sorted(p for p in args.src.iterdir() if p.is_dir()):
        seq_dst = args.dst / seq_src.name
        counts = copy_sequence(seq_src, seq_dst, args.stride, args.copy)
        summaries.append((seq_src.name, counts))

    manifest = args.dst / "SPARSE_EVERY4_MANIFEST.txt"
    with manifest.open("w") as f:
        f.write(f"source: {args.src}\n")
        f.write(f"stride: {args.stride}\n")
        f.write("selection: zero-based frame index % stride == 0\n")
        f.write("frame_assets: hard links unless --copy was used or linking failed\n\n")
        for seq_name, counts in summaries:
            f.write(f"{seq_name}\n")
            for key, value in sorted(counts.items()):
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    for seq_name, counts in summaries:
        print(seq_name)
        for key, value in sorted(counts.items()):
            print(f"  {key}: {value}")
    print(f"\nWrote {args.dst}")


if __name__ == "__main__":
    main()
