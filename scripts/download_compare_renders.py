#!/usr/bin/env python3
"""Download compare renders from Modal with per-file verification."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

MIN_BYTES = 10_000
VOLUME = "nerf-outputs"


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def list_modal_experiments(tag: str) -> list[str]:
    remote = f"compare_renders/{tag}"
    out = _run(["modal", "volume", "ls", VOLUME, remote])
    prefix = f"{remote}/"
    exps = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith(prefix):
            continue
        name = line.removeprefix(prefix).split("/")[0]
        if name.startswith(("bicycle_sparse", "kitti_seq02")) and name not in exps:
            exps.append(name)
    return sorted(exps)


def list_modal_views(tag: str, exp: str) -> list[str]:
    remote = f"compare_renders/{tag}/{exp}"
    out = _run(["modal", "volume", "ls", VOLUME, remote])
    prefix = f"{remote}/"
    views = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith(prefix):
            continue
        part = line.removeprefix(prefix).split("/")[0]
        if part.endswith(".json"):
            continue
        if part not in views:
            views.append(part)
    return sorted(views)


def download_file(remote: str, local: Path, *, min_bytes: int = MIN_BYTES) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists() and local.stat().st_size >= min_bytes:
        return True
    if local.exists():
        local.unlink()
    subprocess.run(
        ["modal", "volume", "get", VOLUME, remote, str(local)],
        check=True,
        capture_output=True,
        text=True,
    )
    ok = local.exists() and local.stat().st_size >= min_bytes
    if not ok and local.exists():
        local.unlink()
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="random5_seed42")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("local_outputs/compare_renders/random5_seed42_all"),
    )
    args = parser.parse_args()

    exps = list_modal_experiments(args.tag)
    print(f"Found {len(exps)} experiments on Modal")
    args.out.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    ok_count = 0
    for exp in exps:
        views = list_modal_views(args.tag, exp)
        exp_dir = args.out / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        for view in views:
            for kind in ("render.png", "gt.png"):
                remote = f"compare_renders/{args.tag}/{exp}/{view}/{kind}"
                local = exp_dir / view / kind
                if download_file(remote, local):
                    ok_count += 1
                else:
                    failed.append(remote)
        manifest_remote = f"compare_renders/{args.tag}/{exp}/manifest.json"
        download_file(manifest_remote, exp_dir / "manifest.json", min_bytes=100)
        print(f"  {exp} ({len(views)} views)")

    print(f"\nDownloaded {ok_count} image files -> {args.out.resolve()}")
    if failed:
        print(f"FAILED ({len(failed)}):")
        for path in failed[:20]:
            print(f"  {path}")
        raise SystemExit(1)

    grouped = args.out.parent / args.tag / "grouped"
    print(f"\nGroup with:")
    print(f"  python3 scripts/group_compare_renders.py {args.out} --output-dir {grouped}")


if __name__ == "__main__":
    main()
