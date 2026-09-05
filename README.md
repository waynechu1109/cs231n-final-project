# Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction

**CS231N Final Project — Stanford University**
Wayne Chu · Yashasvini Gopalan · Changju Yuan

<div class="links">
  <a href="https://waynechu1109.github.io/cs231n-final-project/"><img src="https://img.shields.io/badge/Project_Page-blue" alt="Project Page">
  <a href="https://arxiv.org/abs/2607.02554v1"><img src="https://img.shields.io/badge/arXiv-2607.02554-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="https://waynechu1109.github.io/slides/cs231n_poster.pdf"><img src="https://img.shields.io/badge/Poster-PDF-1f6feb" alt="Poster"></a>
</div>

---

## Overview

Reconstructing outdoor driving scenes from sparse views is difficult due to the narrow forward-facing trajectory of the camera motion and limited multi-view overlap. Monocular depth estimators can provide dense geometric priors, but their predictions are noisy and not consistently reliable across image regions.

This project applies **photometric-masked monocular depth supervision** for sparse-view outdoor reconstruction. We use [Depth Anything V2 (DA-V2)](https://github.com/DepthAnything/Depth-Anything-V2) as a dense monocular depth prior, calibrate it to metric depth with per-image scale-shift fitting on sparse anchors (LiDAR on KITTI, COLMAP on Bicycle), and apply depth supervision selectively via photometric masks derived from an RGB-only baseline. Since the mask modifies an existing depth loss rather than introducing one, we compare against **global (τ=1.0, unmasked) supervision** as the primary baseline. We evaluate on two representative scene representations: **Mip-NeRF-360** and **Splatfacto (3DGS)**.

For method details, see [docs/method.md](docs/method.md).

## Key Results

**Splatfacto — masked-vs-global on KITTI 00 / 02 / 05.** The mask adds +0.44 to +0.70 dB PSNR over global supervision at tied or better RMSE. The RMSE drop comes from using a depth prior at all; masking is what improves rendering fidelity.

| Seq     | Method                       | PSNR ↑ | SSIM ↑ | LPIPS ↓ | RMSE ↓ |
|---------|------------------------------|--------|--------|---------|--------|
| 02/034  | RGB-only                     | 14.90  | 0.433  | 0.446   | 0.542  |
| 02/034  | Global (τ=1.0)               | 15.49  | 0.448  | 0.434   | 0.101  |
| 02/034  | **Masked (τ=0.18)**          | **15.93** | **0.477** | **0.408** | **0.100** |
| 05/018  | RGB-only                     | 14.89  | 0.521  | 0.493   | 0.807  |
| 05/018  | Global (τ=1.0)               | 15.26  | 0.534  | 0.469   | 0.096  |
| 05/018  | **Masked (τ=0.18)**          | **15.90** | **0.548** | **0.446** | **0.096** |
| 00/027  | RGB-only                     | 14.67  | 0.487  | 0.401   | 0.514  |
| 00/027  | Global (τ=1.0)               | 16.69  | 0.554  | 0.302   | 0.125  |
| 00/027  | **Masked (τ=0.18)**          | **17.39** | **0.571** | **0.288** | **0.114** |

**Mip-NeRF-360 — KITTISeq02.** The optimal setting bypasses the mask entirely (global τ=1.0 gives the best PSNR); the implicit density field is more sensitive to noisy monocular depth.

| Setting                 | PSNR ↑ | SSIM ↑    | LPIPS ↓   | RMSE ↓    |
|-------------------------|--------|-----------|-----------|-----------|
| RGB-only                | 20.378 | **0.601** | **0.409** | **2.703** |
| Masked (τ=0.18, λ=0.10) | 20.384 | 0.594     | 0.416     | 3.532     |
| **Global (τ=1.0, λ=0.15)** | **20.607** | 0.595 | 0.412 | 3.580 |

**Comparison vs LiDAR-supervised baselines (KITTISeq02).** Without any GT depth, our mask outperforms three of four LiDAR-supervised methods.

| Method                              | Depth source | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|-------------------------------------|--------------|--------|--------|---------|
| RGB-only                            | none         | 14.903 | 0.433  | 0.446   |
| DA-V2 depth, no mask (global)       | DA-V2        | 15.494 | 0.448  | 0.434   |
| **Ours (τ=0.18, λ=0.10)**           | DA-V2        | **15.932** | **0.477** | **0.408** |
| DNGaussian                          | GT LiDAR     | 9.98   | 0.303  | 0.710   |
| DepthRegGS                          | GT LiDAR     | 8.71   | 0.229  | 0.737   |
| SparseGS                            | GT LiDAR     | 12.20  | 0.359  | 0.648   |
| DN-Splatter                         | GT LiDAR     | **16.22** | **0.489** | **0.289** |

Full results, per-scene breakdown, and takeaways: [docs/results.md](docs/results.md).

## Reproduce the headline result

One Modal command handles data prep → RGB-only baseline → mask generation → depth-supervised retraining:

```bash
modal run --detach modal_train_splatfacto.py::run_new_seq_experiments \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt"
```

Evaluate:

```bash
modal run modal_train_splatfacto.py::run_eval \
  --kitti-seq-dir "KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt" \
  --lambda-depth 0.10 --photo-mask-threshold 0.18 --masked
```

Per-phase breakdown and local commands: [docs/workflow.md](docs/workflow.md) · [docs/quickstart.md](docs/quickstart.md).

## Repository Layout

```text
.
├── modal_train_splatfacto.py     # Modal entry: splatfacto-da2 training, eval, sweeps
├── modal_train_mipnerf.py        # Modal entry: Mip-NeRF-360 (JAX) training, eval
│
├── scripts/                      # Helper scripts, grouped by phase
│   ├── data_prep/                # COLMAP → nerfstudio, KITTI sparse, DA-V2 align
│   ├── masks/                    # Photometric mask generation (fixed + matched-ratio)
│   ├── train/                    # Splatfacto + Mip-NeRF training shells
│   └── eval/                     # Metric computation
│
├── outdoor-nerf-depth/           # Mip-NeRF-360 backbone (vendored, patched)
├── nerfstudio/                   # Splatfacto backbone (vendored)
├── Depth-Anything-V2/            # DA-V2 inference + scale-shift align (vendored)
│
├── docs/                         # Per-topic guides (see below)
├── index.html                    # Project page
└── media/                        # Project-page figures
```

## Documentation

| Topic | File |
|---|---|
| Method (alignment, masked supervision, ablations) | [docs/method.md](docs/method.md) |
| End-to-end workflow (Phases 1–5) | [docs/workflow.md](docs/workflow.md) |
| Datasets and remote data paths | [docs/datasets.md](docs/datasets.md) |
| Local Quick Start (env setup + 6-step commands) | [docs/quickstart.md](docs/quickstart.md) |
| Results summary and takeaways | [docs/results.md](docs/results.md) |
| Modal cloud training (Splatfacto) | [docs/modal-splatfacto.md](docs/modal-splatfacto.md) |
| Modal cloud training (Mip-NeRF-360) | [docs/modal-mipnerf.md](docs/modal-mipnerf.md) |
| Depth Anything V2 preprocessing | [docs/depth-anything-v2.md](docs/depth-anything-v2.md) |
| Mip-NeRF-360 KITTI setup | [docs/mipnerf360-kitti.md](docs/mipnerf360-kitti.md) |
| Mip-NeRF-360 Bicycle sparse (λ / mask sweep) | [docs/mip360-bicycle-sparse.md](docs/mip360-bicycle-sparse.md) |
| Nerfstudio Splatfacto training | [docs/nerfstudio-splatfacto.md](docs/nerfstudio-splatfacto.md) |
| Nerfstudio Splatfacto DA-V2 depth supervision | [docs/nerfstudio-splatfacto-da2.md](docs/nerfstudio-splatfacto-da2.md) |
| Troubleshooting (common failure modes) | [docs/troubleshooting.md](docs/troubleshooting.md) |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{chu2026reliability,
    title         = {Reliability-Aware Monocular Depth Supervision for Sparse-View Neural Reconstruction},
    author        = {Wei-Teng Chu and Yashasvini Gopalan and Changju Yuan},
    year          = {2026},
    eprint        = {2607.02554},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CV},
    url           = {https://arxiv.org/abs/2607.02554}
}
```

This project uses code and data organization from:

```bibtex
@article{wang2023digging,
    title={Digging into Depth Priors for Outdoor Neural Radiance Fields},
    author={Chen Wang and Jiadai Sun and Lina Liu and Chenming Wu
            and Zhelun Shen and Dayan Wu and Yuchao Dai and Liangjun Zhang},
    journal={Proceedings of the 31th ACM International Conference on Multimedia},
    year={2023}
}
```

## Acknowledgements

This project builds on:

- [Digging into Depth Priors for Outdoor Neural Radiance Fields](https://github.com/barbararoessle/e2e_multi_view_stereo) (primary baseline)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)
- [Nerfstudio / Splatfacto](https://docs.nerf.studio/)
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
