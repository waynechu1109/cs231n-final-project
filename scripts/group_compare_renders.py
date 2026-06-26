#!/usr/bin/env python3
"""Reorganize compare renders: view / threshold / lambda."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

EXP_RE = re.compile(
    r"^(?:bicycle_sparse_da2|kitti_seq02_0034_sparse_every2_da2)_lambda"
    r"(?P<lambda>[\d.]+)_(?P<threshold>nomask|low\d+|high\d+)_\d+$"
)


def parse_exp_name(exp_name: str) -> tuple[str, str]:
    match = EXP_RE.match(exp_name)
    if not match:
        raise ValueError(f"Cannot parse experiment name: {exp_name}")
    lam = match.group("lambda")
    threshold = match.group("threshold")
    return f"lambda{lam}", threshold


def find_experiment_roots(compare_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for path in compare_dir.rglob("manifest.json"):
        roots.append(path.parent)
    if roots:
        return sorted(set(roots))
    for path in compare_dir.iterdir():
        if path.is_dir() and EXP_RE.match(path.name):
            roots.append(path)
    return sorted(roots)


def group_renders(compare_dir: Path, output_dir: Path | None = None, *, move: bool = False) -> dict:
    compare_dir = compare_dir.resolve()
    out = output_dir.resolve() if output_dir else compare_dir
    out.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {"views": {}, "experiments": []}
    exp_roots = find_experiment_roots(compare_dir)

    for exp_root in exp_roots:
        exp_name = exp_root.name
        lambda_dir, threshold = parse_exp_name(exp_name)
        index["experiments"].append(exp_name)

        for view_dir in sorted(exp_root.iterdir()):
            if not view_dir.is_dir() or view_dir.name.startswith("."):
                continue
            view_name = view_dir.name
            render_src = view_dir / "render.png"
            gt_src = view_dir / "gt.png"
            if not render_src.exists():
                continue

            dest = out / view_name / threshold / lambda_dir
            dest.mkdir(parents=True, exist_ok=True)

            op = shutil.move if move else shutil.copy2
            op(render_src, dest / "render.png")
            if gt_src.exists():
                gt_dest = out / view_name / "gt.png"
                if not gt_dest.exists():
                    op(gt_src, gt_dest)

            index["views"].setdefault(view_name, {}).setdefault(threshold, []).append(lambda_dir)

    for view in index["views"]:
        for threshold in index["views"][view]:
            index["views"][view][threshold] = sorted(set(index["views"][view][threshold]))

    index_path = out / "grouped_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group compare renders by view/threshold/lambda.")
    parser.add_argument(
        "compare_dir",
        type=Path,
        help="Compare output dir (contains experiment folders or nested random5_seed42/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination (default: <compare_dir>/by_view).",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir or (args.compare_dir / "by_view")
    index = group_renders(args.compare_dir, out, move=args.move)
    print(f"Grouped renders -> {out.resolve()}")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
