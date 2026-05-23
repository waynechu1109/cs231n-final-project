"""
Full-image dataset that loads per-frame depth maps for Gaussian splatting supervision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path


class DepthFullImageDataset(InputDataset):
    """Dataset that returns images and depth maps for full-image trainers such as splatfacto."""

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["depth_image"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.depth_filenames = self.metadata.get("depth_filenames")
        self.depth_unit_scale_factor = self.metadata.get("depth_unit_scale_factor", 1e-3)
        self.metadata["dataparser_scale"] = self._dataparser_outputs.dataparser_scale

    def get_metadata(self, data: Dict) -> Dict:
        if self.depth_filenames is None:
            return {}

        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
        depth_image = get_depth_image_from_path(
            filepath=Path(filepath),
            height=height,
            width=width,
            scale_factor=self.depth_unit_scale_factor,
        )
        depth_crop_range = float(self.metadata.get("depth_crop_range", 0.0))
        if depth_crop_range > 0.0:
            depth_image[depth_image > depth_crop_range] = 0.0
        depth_image = depth_image * self._dataparser_outputs.dataparser_scale
        return {"depth_image": depth_image}
