#!/usr/bin/env python3
"""Add train/val/test filename lists to transforms.json (KITTI-style holdout)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch transforms.json with train_filenames / val_filenames / test_filenames "
            "using the same sparse holdout as scripts/make_nerfstudio_kitti_sparse.py."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Nerfstudio dataset directory containing transforms.json.",
    )
    parser.add_argument(
        "--hold-every",
        type=int,
        default=10,
        help="Hold out every Nth frame (0-based index N-1) for test.",
    )
    parser.add_argument(
        "--val-offset",
        type=int,
        default=None,
        help="Sparse-frame offset for val (default: hold_every // 2 - 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.hold_every <= 0:
        raise SystemExit("--hold-every must be positive")
    val_offset = args.val_offset
    if val_offset is None:
        val_offset = args.hold_every // 2 - 1
    if val_offset < 0 or val_offset >= args.hold_every:
        raise SystemExit("--val-offset must be in [0, hold_every)")
    if val_offset == args.hold_every - 1:
        raise SystemExit("--val-offset must not overlap test offset")

    transforms_path = args.data_dir / "transforms.json"
    if not transforms_path.exists():
        raise SystemExit(f"Missing {transforms_path}")

    with transforms_path.open() as f:
        meta = json.load(f)

    frames = sorted(meta["frames"], key=lambda fr: Path(fr["file_path"]).name)
    if not frames:
        raise SystemExit("No frames in transforms.json")

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

    meta["frames"] = frames
    meta["train_filenames"] = train
    meta["val_filenames"] = val
    meta["test_filenames"] = test

    with transforms_path.open("w") as f:
        json.dump(meta, f, indent=4)
        f.write("\n")

    print(f"Patched {transforms_path}")
    print(f"frames: {len(frames)}  train: {len(train)}  val: {len(val)}  test: {len(test)}")


if __name__ == "__main__":
    main()
