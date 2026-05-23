"""
Splatfacto augmented with depth supervised training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

import torch

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.utils import colormaps


@dataclass
class SplatfactoDepthModelConfig(SplatfactoModelConfig):
    """Additional parameters for depth supervision during Gaussian splatting."""

    _target: Type = field(default_factory=lambda: SplatfactoDepthModel)
    lambda_depth: float = 0.05
    """Weight of the depth loss, matching MipNeRF-360 Config.lambda_depth."""
    depth_loss_type: Literal["mse", "l1"] = "mse"
    """Depth loss type, matching MipNeRF-360 Config.depth_loss_type."""
    depth_min_valid: float = 1.0 / 256.0
    """Ignore supervision depth below this value in meters (KITTI invalid pixels use raw < 2)."""
    output_depth_during_training: bool = True
    """Must be True so rendered depth is available for the depth loss."""


class SplatfactoDepthModel(SplatfactoModel):
    """Depth-supervised splatfacto model."""

    config: SplatfactoDepthModelConfig

    def _get_depth_supervision_mask(self, depth_gt: torch.Tensor) -> torch.Tensor:
        min_valid = self.config.depth_min_valid
        if hasattr(self, "metadata") and self.metadata is not None:
            dataparser_scale = float(self.metadata.get("dataparser_scale", 1.0))
            min_valid = self.config.depth_min_valid * dataparser_scale
        return depth_gt[..., 0] > min_valid

    def _compute_depth_loss(self, pred_depth: torch.Tensor, depth_gt: torch.Tensor) -> torch.Tensor:
        mask = self._get_depth_supervision_mask(depth_gt)
        if not torch.any(mask):
            return pred_depth.new_zeros(())

        pred = pred_depth[mask]
        gt = depth_gt[mask]
        if self.config.depth_loss_type == "mse":
            return torch.mean((pred - gt) ** 2)
        if self.config.depth_loss_type == "l1":
            return torch.mean(torch.abs(pred - gt))
        raise ValueError(f"Unknown depth loss type: {self.config.depth_loss_type}")

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training and "depth_image" in batch and outputs.get("depth") is not None:
            depth_gt = self._downscale_if_required(batch["depth_image"].to(self.device))
            pred_depth = outputs["depth"]
            if pred_depth.dim() == 3 and pred_depth.shape[-1] == 1:
                pred_depth = pred_depth[..., 0]
            if depth_gt.dim() == 3 and depth_gt.shape[-1] == 1:
                depth_gt = depth_gt[..., 0]
            metrics_dict["depth_loss"] = self._compute_depth_loss(pred_depth, depth_gt)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and metrics_dict is not None and "depth_loss" in metrics_dict:
            loss_dict["depth_loss"] = self.config.lambda_depth * metrics_dict["depth_loss"]
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics, images = super().get_image_metrics_and_images(outputs, batch)
        if "depth_image" not in batch or outputs.get("depth") is None:
            return metrics, images

        ground_truth_depth = batch["depth_image"].to(self.device)
        if ground_truth_depth.dim() == 2:
            ground_truth_depth = ground_truth_depth.unsqueeze(-1)

        ground_truth_depth_colormap = colormaps.apply_depth_colormap(ground_truth_depth)
        valid = self._get_depth_supervision_mask(ground_truth_depth)
        if torch.any(valid):
            near_plane = float(torch.min(ground_truth_depth[valid]).cpu())
            far_plane = float(torch.max(ground_truth_depth[valid]).cpu())
        else:
            near_plane = 0.0
            far_plane = 1.0
        predicted_depth_colormap = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=near_plane,
            far_plane=far_plane,
        )
        images["depth"] = torch.cat([ground_truth_depth_colormap, predicted_depth_colormap], dim=1)

        if torch.any(valid):
            pred = outputs["depth"][..., 0][valid]
            gt = ground_truth_depth[..., 0][valid]
            metrics["depth_mse"] = float(torch.mean((pred - gt) ** 2).cpu())
            metrics["depth_mae"] = float(torch.mean(torch.abs(pred - gt)).cpu())
        return metrics, images
