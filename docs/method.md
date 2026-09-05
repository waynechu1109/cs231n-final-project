# Method

Photometric-masked monocular depth supervision for sparse-view outdoor scene reconstruction. This document details the alignment procedure, the reliability mask, the training objective, and the ablations that validate the design.

## Monocular Depth Prior Alignment

DA-V2 predicts relative depth. We align each prediction to metric depth via per-image least-squares scale-shift fitting — solving for the scale *s* and shift *t* that best match the predicted depth *d_m* to the reference depth *d_r* over valid anchor pixels.

For KITTI, reference depth comes from projected LiDAR points. For Mip-NeRF-360 Bicycle, sparse COLMAP points are used. Aligned depth is clipped to 80 m for KITTI.

## Photometric-Masked Depth Supervision

Instead of applying the depth loss everywhere, we only supervise the pixels where the RGB-only baseline is already reliable. A pre-trained RGB-only model is used to compute a per-pixel photometric error map:

$$e(u) = \frac{1}{3} \sum_{c \in \lbrace R,G,B \rbrace} \left| \hat{I}_c(u) - I_c(u) \right|$$

Pixels with $e(u) < \tau$ form the binary reliability mask $M(u)$. This mask is fixed for the duration of depth-supervised training.

## Training Pipeline

1. **Stage 1** — Train RGB-only baseline for 50 000 iterations.
2. **Stage 2** — Render all training views; compute per-pixel photometric error; generate fixed masks at threshold τ.
3. **Stage 3** — Retrain from a fresh initialization with depth supervision: $\mathcal{L} = \mathcal{L}_\text{rgb} + \lambda_\text{depth}\,\mathcal{L}_\text{depth}$, where $\mathcal{L}_\text{depth}$ is MSE gated by $M_\text{eff}(u) = M(u) \land D(u)$ ($D$ = depth-validity mask).

We sweep τ ∈ {0.14, 0.16, 0.18, 0.20, 0.22, 1.0} and λ ∈ {0.05, 0.10, 0.15}. **Setting τ = 1.0 gives $M \equiv 1$ — global (unmasked) depth supervision** — which is our primary baseline for judging the mask's contribution, since it isolates the effect of *where* depth is supervised from *whether* depth is supervised at all.

## Matched-Ratio Ablation

To test whether the benefit of low-error masking comes from **selecting reliable pixels** vs simply **using fewer pixels**, we construct two control masks that match the exact pixel budget of `low018` (i.e. the same number of supervised pixels per frame):

- **High-error matched** — selects the *k* valid-depth pixels with the **highest** photometric error per frame.
- **Random matched** — selects *k* valid-depth pixels **uniformly at random** per frame (three seeds for stability).

where *k* = |{u : valid_depth(u) ∧ e(u) < 0.18}| per frame, computed from the RGB-only baseline. All ablation runs use λ = 0.10, τ = 0.18 on KITTISeq02 every-2. The low-error mask wins on every metric — the improvement is not from fewer supervised pixels.

| Mask (λ=0.10)                    | PSNR ↑ | SSIM ↑ | LPIPS ↓ | RMSE ↓ |
|----------------------------------|--------|--------|---------|--------|
| High-error, matched              | 14.932 | 0.437  | 0.455   | 0.111  |
| Random, matched (3 seeds)        | 15.036 | 0.442  | 0.456   | 0.109  |
| **Low-error, τ=0.18 (ours)**     | **15.932** | **0.477** | **0.408** | **0.100** |

## Mask Validity Against GT LiDAR

Because photometric and depth errors are measured against different references, low photometric error does not by construction imply low depth error. We validate the mask directly against GT LiDAR. Depth RMSE is **34–39% lower inside the mask** at every threshold (6.33 vs 10.01 m at τ=0.18), and inside-mask RMSE increases monotonically with τ (r=1.0). Pixel-level differences are highly significant (p < 10⁻⁵⁰); frame-level paired t-test gives p = 0.047 (n = 8).

## Comparison to a Depth-Inconsistency Mask

We also compare against a static, one-shot approximation of the Depth-Inconsistency Mask (DIM) at a matched reliable-pixel fraction. Ratio = MAE(outside) / MAE(inside): >1 means the mask isolates accurate DA-V2 depth. Our mask separates accurate from inaccurate depth (ratio **1.17**) while the DIM proxy does not (**0.87**): DIM keys on distortions that depth-supervised training itself induces, so a one-shot proxy has nothing to key on.

| Mask                | Threshold | Fraction | MAE in / out (m) | Ratio |
|---------------------|-----------|----------|------------------|-------|
| **Photometric (ours)** | τ = 0.18 | 95.9%   | 4.14 / 4.82     | **1.17** |
| DIM proxy           | ε = 17.6 m | 96.1%  | 4.19 / 3.66     | 0.87  |
