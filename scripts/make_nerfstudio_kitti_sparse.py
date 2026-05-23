#!/usr/bin/env python3
"""Create a Nerfstudio KITTI sparse dataset from an existing transforms.json."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a Nerfstudio KITTI transforms.json to every Nth frame."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/nerfstudio/kitti_seq02_0034"),
        help="Source dense Nerfstudio dataset.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/nerfstudio/kitti_seq02_0034_sparse_every4"),
        help="Output sparse Nerfstudio dataset.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "data/kitti/kitti_select_static_5seq_sparse_every4/"
            "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt/images"
        ),
        help="Sparse image directory to link as dst/images.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Keep frames whose original numeric filename is divisible by this value.",
    )
    parser.add_argument(
        "--hold-every",
        type=int,
        default=10,
        help="Hold out every Nth sparse frame for test, starting at sparse index N-1.",
    )
    parser.add_argument(
        "--val-offset",
        type=int,
        default=None,
        help=(
            "Sparse-frame offset for validation frames. Defaults to the midpoint "
            "between test frames, e.g. 4 when --hold-every=10."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination dataset if it already exists.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy the sparse images instead of creating a symlink.",
    )
    return parser.parse_args()


def frame_number(frame: dict) -> int:
    return int(Path(frame["file_path"]).stem)


def make_relative_to_cwd(path: Path) -> Path:
    try:
        return path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path.resolve()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if args.hold_every <= 0:
        raise SystemExit("--hold-every must be positive")
    val_offset = args.val_offset
    if val_offset is None:
        val_offset = args.hold_every // 2 - 1
    if val_offset < 0 or val_offset >= args.hold_every:
        raise SystemExit("--val-offset must be in [0, hold_every)")
    if val_offset == args.hold_every - 1:
        raise SystemExit("--val-offset must not overlap the test offset")
    if args.dst.exists():
        if not args.overwrite:
            raise SystemExit(f"Destination already exists: {args.dst}")
        if args.dst.is_symlink() or args.dst.is_file():
            args.dst.unlink()
        else:
            shutil.rmtree(args.dst)

    transforms_path = args.src / "transforms.json"
    if not transforms_path.exists():
        raise SystemExit(f"Missing source transforms.json: {transforms_path}")
    if not args.images.exists():
        raise SystemExit(f"Missing sparse image directory: {args.images}")

    with transforms_path.open() as f:
        meta = json.load(f)

    frames = [frame for frame in meta["frames"] if frame_number(frame) % args.stride == 0]
    if not frames:
        raise SystemExit("No frames selected")

    available = {p.name for p in args.images.iterdir() if p.is_file()}
    missing = [Path(frame["file_path"]).name for frame in frames if Path(frame["file_path"]).name not in available]
    if missing:
        preview = ", ".join(missing[:5])
        raise SystemExit(f"{len(missing)} selected frames are missing from sparse images: {preview}")

    test_held_out = {
        frame["file_path"]
        for idx, frame in enumerate(frames)
        if idx % args.hold_every == args.hold_every - 1
    }
    val_held_out = {
        frame["file_path"]
        for idx, frame in enumerate(frames)
        if idx % args.hold_every == val_offset
    }
    held_out = test_held_out | val_held_out
    train = [frame["file_path"] for frame in frames if frame["file_path"] not in held_out]
    val = [frame["file_path"] for frame in frames if frame["file_path"] in val_held_out]
    test = [frame["file_path"] for frame in frames if frame["file_path"] in test_held_out]

    out_meta = dict(meta)
    out_meta["frames"] = frames
    out_meta["train_filenames"] = train
    out_meta["val_filenames"] = val
    out_meta["test_filenames"] = test

    args.dst.mkdir(parents=True)
    if args.copy_images:
        shutil.copytree(args.images, args.dst / "images")
    else:
        os.symlink(args.images.resolve(), args.dst / "images")

    with (args.dst / "transforms.json").open("w") as f:
        json.dump(out_meta, f, indent=4)
        f.write("\n")

    with (args.dst / f"SPARSE_EVERY{args.stride}_MANIFEST.txt").open("w") as f:
        f.write(f"source_nerfstudio: {make_relative_to_cwd(args.src)}\n")
        f.write(f"sparse_images: {make_relative_to_cwd(args.images)}\n")
        f.write(f"stride: {args.stride}\n")
        f.write(f"hold_every_sparse_frame: {args.hold_every}\n")
        f.write(f"test_sparse_offset: {args.hold_every - 1}\n")
        f.write(f"val_sparse_offset: {val_offset}\n")
        f.write(f"frames: {len(frames)}\n")
        f.write(f"train: {len(train)}\n")
        f.write(f"val: {len(val)}\n")
        f.write(f"test: {len(test)}\n")
        f.write("\nval_filenames:\n")
        for name in val:
            f.write(f"  {name}\n")
        f.write("\ntest_filenames:\n")
        for name in test:
            f.write(f"  {name}\n")

    print(f"Wrote {args.dst}")
    print(f"frames: {len(frames)}")
    print(f"train: {len(train)}")
    print(f"val: {len(val)}")
    print(f"test: {len(test)}")
    print("val filenames:")
    for name in val:
        print(f"  {name}")
    print("test filenames:")
    for name in test:
        print(f"  {name}")


if __name__ == "__main__":
    main()
