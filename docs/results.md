# Results Summary

Full results and paper's headline numbers. See the top-level [README](../README.md) for the compact result tables.

## Splatfacto — KITTI 00 / 02 / 05 (Sparse Every-2)

Splatfacto benefits from monocular depth supervision. Using any DA-V2 depth prior — even without a mask — cuts RMSE by ~80% (KITTISeq02: 0.542 → 0.101). Masking the depth loss adds a further **+0.44 to +0.70 dB PSNR** over global supervision at tied or better RMSE across sequences 00 / 02 / 05. Over three seeds on KITTISeq02: masked 15.30 ± 0.57 vs global 15.07 ± 0.38 dB (mean +0.23 dB). Bottom line: the RMSE drop comes from using a depth prior at all, and masking is what improves rendering fidelity. Increasing λ from 0.10 to 0.15 at τ=0.18 improves RMSE (0.100 → 0.096) but reduces PSNR (15.932 → 15.588), showing a geometry–photometry tradeoff.

## Mip-NeRF-360 — KITTI 02 and 05 (Sparse Every-2)

The implicit density field is far more sensitive to noisy monocular depth. On KITTISeq02 the optimal setting is global (τ=1.0) — the mask brings no gain; all depth-supervised settings degrade SSIM, LPIPS, and geometry relative to RGB-only. On KITTISeq05 both global and masked supervision degrade rendering and geometry vs RGB-only; masking recovers 0.22 dB PSNR over global (16.612 vs 16.389), so it mitigates but does not reverse the harm.

| Seq | Setting                | PSNR ↑ | SSIM ↑    | LPIPS ↓   | AbsRel ↓ | RMSE ↓    |
|-----|------------------------|--------|-----------|-----------|----------|-----------|
| 05  | **RGB-only**           | **17.068** | **0.546** | **0.529** | **0.1166** | **2.978** |
| 05  | Global (τ=1.0, λ=0.15) | 16.389 | 0.527     | 0.569     | 0.1527   | 4.803     |
| 05  | Masked (τ=0.18, λ=0.15)| 16.612 | 0.530     | 0.563     | 0.1399   | 4.454     |

## Mip-NeRF-360 Bicycle — Circular Sparse Views

RGB-only Splatfacto achieves the best rendering quality (17.731 PSNR). Depth supervision consistently reduces RMSE (1.479 → 0.722) but degrades PSNR/SSIM/LPIPS. Global supervision gives the best PSNR among depth-supervised runs (17.593 vs 17.466 for the best mask), confirming the mask's rendering gain is specific to sparse forward-facing trajectories — depth regularization over-constrains an already well-posed reconstruction.

## Depth Prior Quality

Average scale-shift alignment error on KITTI LiDAR: **4.22 m**. Sweeping the number of anchors used for scale-shift fitting shows the 2-DOF optimization saturates well before typical LiDAR density: MAE is stable from ~95k anchors down to 500 per frame, only degrading under extreme sparsity (4.29 m at 100 anchors, 4.51 m at 20). The 4.22 m error floor reflects DA-V2's intrinsic local structural limitations, not calibration artifacts. DA-V2 is most reliable on well-textured mid-range surfaces and noisiest on reflective surfaces, thin structures, and distant regions.

## Takeaways

Monocular depth priors are most useful for **explicit Gaussian representations** in under-constrained, forward-facing sparse-view scenes, and less reliable for implicit NeRF-style density fields. The mask's contribution is rendering fidelity, not metric geometry: +0.44 to +0.70 dB PSNR over global supervision on KITTI 00/02/05 at tied or better RMSE, no gain for Mip-NeRF-360 or the object-centric Bicycle scene. Without any ground-truth depth, our mask also outperforms three LiDAR-supervised baselines. The explicit-vs-implicit architectural insights are prior-agnostic; sweep values (τ, λ) are dataset-specific.
