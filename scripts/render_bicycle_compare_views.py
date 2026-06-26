#!/usr/bin/env python3
"""Render selected dataset camera views from a splatfacto-da2 checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NERFSTUDIO_ROOT = PROJECT_ROOT / "nerfstudio"
sys.path.insert(0, str(NERFSTUDIO_ROOT))

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig  # noqa: E402
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig  # noqa: E402
from nerfstudio.engine.trainer import TrainerConfig  # noqa: E402
from nerfstudio.utils.eval_utils import eval_setup  # noqa: E402


def _test_filenames(transforms_path: Path) -> list[str]:
    meta = json.loads(transforms_path.read_text())
    if "test_filenames" in meta:
        return [Path(p).name for p in meta["test_filenames"]]
    frames = sorted(meta["frames"], key=lambda fr: Path(fr["file_path"]).name)
    hold_every = 10
    return [
        Path(frame["file_path"]).name
        for idx, frame in enumerate(frames)
        if idx % hold_every == hold_every - 1
    ]


def pick_random_test_views(
    transforms_path: Path,
    num_views: int,
    seed: int,
) -> list[dict[str, Any]]:
    names = _test_filenames(transforms_path)
    if num_views > len(names):
        raise ValueError(f"Requested {num_views} views but only {len(names)} test frames exist.")
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(names)), num_views))
    return [
        {"split": "test", "index": idx, "filename": names[idx]}
        for idx in indices
    ]


def _tensor_to_uint8_rgb(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def _get_split_dataset(pipeline, split: str):
    data_manager_config = pipeline.datamanager.config
    assert isinstance(data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
    if split == "train":
        return pipeline.datamanager.train_dataset
    datamanager = data_manager_config.setup(test_mode=split, device=pipeline.device)
    return datamanager.eval_dataset


def _depth_to_uint8(depth_tensor: torch.Tensor) -> np.ndarray | None:
    depth = np.squeeze(depth_tensor.detach().cpu().numpy())
    if depth.ndim != 2 or depth.size == 0:
        return None
    depth_norm = depth - np.nanmin(depth)
    denom = np.nanmax(depth_norm)
    if denom > 0:
        depth_norm = depth_norm / denom
    return (depth_norm * 255).clip(0, 255).astype(np.uint8)


def render_views_for_config(
    load_config: Path,
    views: list[dict[str, Any]],
    output_dir: Path,
    *,
    rendered_outputs: tuple[str, ...] = ("rgb",),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _update_config(config: TrainerConfig) -> TrainerConfig:
        dm = config.pipeline.datamanager
        assert isinstance(dm, (VanillaDataManagerConfig, FullImageDatamanagerConfig))
        dm.eval_num_images_to_sample_from = -1
        dm.eval_num_times_to_repeat_images = -1
        if isinstance(dm, VanillaDataManagerConfig):
            dm.train_num_images_to_sample_from = -1
            dm.train_num_times_to_repeat_images = -1
        return config

    _config, pipeline, checkpoint_path, step = eval_setup(
        load_config,
        test_mode="inference",
        update_config_callback=_update_config,
    )
    pipeline.eval()
    pipeline.model.eval()
    device = pipeline.device

    manifest: dict[str, Any] = {
        "load_config": str(load_config),
        "checkpoint": str(checkpoint_path),
        "step": int(step),
        "views": views,
        "outputs": {},
    }

    with torch.no_grad():
        for view in views:
            split = view.get("split", "test")
            idx = int(view["index"])
            filename = view.get("filename")
            dataset = _get_split_dataset(pipeline, split)
            if idx < 0 or idx >= len(dataset):
                raise IndexError(f"View index {idx} out of range for split={split} (size={len(dataset)})")

            batch = dataset.get_data(idx, image_type="float32")
            camera = dataset.cameras[idx : idx + 1].to(device)
            outputs = pipeline.model.get_outputs_for_camera(camera)
            if filename is None:
                filename = dataset.image_filenames[idx].name

            view_dir = output_dir / Path(filename).stem
            view_dir.mkdir(parents=True, exist_ok=True)
            view_record: dict[str, str] = {"filename": filename, "split": split, "index": idx}

            gt_rgb = _tensor_to_uint8_rgb(batch["image"])
            Image.fromarray(gt_rgb).save(view_dir / "gt.png")
            view_record["gt"] = str(view_dir / "gt.png")

            if "rgb" in rendered_outputs:
                pred_rgb = _tensor_to_uint8_rgb(outputs["rgb"])
                Image.fromarray(pred_rgb).save(view_dir / "render.png")
                view_record["render"] = str(view_dir / "render.png")

            if "depth" in rendered_outputs and "depth" in outputs:
                depth_u8 = _depth_to_uint8(outputs["depth"])
                if depth_u8 is not None:
                    Image.fromarray(depth_u8).save(view_dir / "depth.png")
                    view_record["depth"] = str(view_dir / "depth.png")

            manifest["outputs"][filename] = view_record

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render selected views from a splatfacto-da2 checkpoint.")
    parser.add_argument("--load-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views-json", type=Path, default=None, help="JSON list of {split,index,filename}.")
    parser.add_argument("--transforms-json", type=Path, default=None, help="Used to pick random test views.")
    parser.add_argument("--num-views", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.views_json is not None:
        views = json.loads(args.views_json.read_text())
    else:
        transforms_path = args.transforms_json or (
            PROJECT_ROOT / "data/nerfstudio/bicycle_sparse/transforms.json"
        )
        views = pick_random_test_views(transforms_path, args.num_views, args.seed)
        views_path = args.output_dir / "selected_views.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        views_path.write_text(json.dumps(views, indent=2) + "\n")
        print(f"Selected views written to {views_path}")

    manifest = render_views_for_config(args.load_config, views, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
