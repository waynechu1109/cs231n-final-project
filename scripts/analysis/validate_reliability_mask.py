#!/usr/bin/env python3
"""
Validate the photometric-error reliability mask against LiDAR ground truth.

Computes per-pixel DA-V2 depth error (AbsRel, RMSE) inside vs. outside the
mask, and Pearson/Spearman correlation between photometric error bins and
depth error, using only locally saved files (no model re-rendering needed).

Usage (from project root):
  conda run -n cs231n-final python scripts/analysis/validate_reliability_mask.py \
    --seq kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt \
    --inside-mask-threshold 0.18
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import stats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_kitti_depth(path: Path) -> np.ndarray:
    """Read KITTI-style 16-bit PNG depth (value / 256 = metres, 0 = invalid)."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    depth = raw.astype(np.float32) / 256.0
    depth[raw == 0] = np.nan
    return depth


def read_mask(path: Path) -> np.ndarray:
    """Read binary mask PNG; returns bool array (True = inside / keep pixel)."""
    img = np.array(Image.open(path))
    return img > 127


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def compute_depth_metrics(da2: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> dict:
    """Return AbsRel and RMSE for pixels where mask is True and gt is valid."""
    valid = mask & np.isfinite(gt) & np.isfinite(da2) & (gt > 0)
    if valid.sum() == 0:
        return {"absrel": np.nan, "rmse": np.nan, "n": 0}
    err = np.abs(da2[valid] - gt[valid])
    absrel = float(np.mean(err / gt[valid]))
    rmse = float(np.sqrt(np.mean((da2[valid] - gt[valid]) ** 2)))
    return {"absrel": absrel, "rmse": rmse, "n": int(valid.sum())}


def collect_pixel_arrays(seq_dir: Path, mask_dir: Path):
    """
    Return flat arrays of DA-V2 depth error and photometric-error bin label
    for every valid pixel (LiDAR valid) that appears in the mask files.

    Returns (da2_vals, gt_vals, mask_bool) arrays over all frames.
    """
    da2_dir = seq_dir / "depths_da2"
    gt_dir  = seq_dir / "depths_gt"

    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        raise FileNotFoundError(f"No PNG files in {mask_dir}")

    all_da2, all_gt, all_mask = [], [], []
    for mf in mask_files:
        stem = mf.stem
        da2_path = da2_dir / f"{stem}.png"
        gt_path  = gt_dir  / f"{stem}.png"
        if not da2_path.exists() or not gt_path.exists():
            print(f"  [skip] missing DA2 or GT for {stem}")
            continue
        da2  = read_kitti_depth(da2_path)
        gt   = read_kitti_depth(gt_path)
        mask = read_mask(mf)
        # Resize mask if needed (DA2 and GT may differ from mask resolution)
        if mask.shape != gt.shape:
            mask = cv2.resize(mask.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        all_da2.append(da2)
        all_gt.append(gt)
        all_mask.append(mask)

    return (np.concatenate([a.ravel() for a in all_da2]),
            np.concatenate([a.ravel() for a in all_gt]),
            np.concatenate([a.ravel() for a in all_mask]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "kitti",
        help="Root directory containing KITTI sequence folders."
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt",
        help="Relative path from --data-root to the sequence directory."
    )
    parser.add_argument(
        "--inside-mask-threshold", "-t",
        type=float,
        default=0.18,
        help="The photometric error threshold τ defining the 'inside' mask (e.g. 0.18)."
    )
    parser.add_argument(
        "--max-depth", type=float, default=80.0,
        help="Clip depth values above this (metres) as invalid."
    )
    args = parser.parse_args()

    seq_dir = args.data_root / args.seq
    if not seq_dir.exists():
        sys.exit(f"Sequence directory not found: {seq_dir}")

    # ------------------------------------------------------------------
    # 1. Find the mask directory closest to the requested threshold
    # ------------------------------------------------------------------
    thresh_tag = f"{args.inside_mask_threshold:.2f}".replace(".", "")  # "018"
    exact_mask_dir = seq_dir / f"photo_masks_rgbonly_low{thresh_tag}_sampleevery2"

    if exact_mask_dir.exists():
        chosen_thresh = args.inside_mask_threshold
        chosen_mask_dir = exact_mask_dir
        print(f"Using exact mask: {chosen_mask_dir.name}")
    else:
        # Find nearest available threshold
        candidates = []
        for d in seq_dir.glob("photo_masks_rgbonly_low*_sampleevery2"):
            tag = d.name.replace("photo_masks_rgbonly_low", "").replace("_sampleevery2", "")
            try:
                t = float(tag) / (10 ** (len(tag) - 1))
                candidates.append((t, d))
            except ValueError:
                pass
        if not candidates:
            sys.exit("No photo_masks_rgbonly_low* directories found.")
        candidates.sort(key=lambda x: abs(x[0] - args.inside_mask_threshold))
        chosen_thresh, chosen_mask_dir = candidates[0]
        print(f"[WARNING] Exact τ={args.inside_mask_threshold} mask not found locally.")
        print(f"          Using nearest available: τ={chosen_thresh} → {chosen_mask_dir.name}")

    # ------------------------------------------------------------------
    # 2. Inside vs. outside mask analysis
    # ------------------------------------------------------------------
    print(f"\n=== Inside vs. Outside Mask Analysis (τ={chosen_thresh}) ===")
    da2_flat, gt_flat, mask_flat = collect_pixel_arrays(seq_dir, chosen_mask_dir)

    # Validity: LiDAR GT must be present and within depth range
    valid = np.isfinite(gt_flat) & np.isfinite(da2_flat) & (gt_flat > 0.5) & (gt_flat < args.max_depth)
    da2_clip = np.clip(da2_flat, 0, args.max_depth)

    inside  = valid & mask_flat
    outside = valid & ~mask_flat

    def metrics(da2, gt, sel):
        err = np.abs(da2[sel] - gt[sel])
        return {
            "n":      int(sel.sum()),
            "absrel": float(np.mean(err / gt[sel])),
            "rmse":   float(np.sqrt(np.mean((da2[sel] - gt[sel]) ** 2))),
            "mae":    float(np.mean(err)),
        }

    m_in  = metrics(da2_clip, gt_flat, inside)
    m_out = metrics(da2_clip, gt_flat, outside)

    print(f"\n{'':30s} {'AbsRel':>8}  {'RMSE':>8}  {'MAE':>8}  {'N pixels':>10}")
    print(f"{'Inside mask (e < τ)':30s} {m_in['absrel']:8.4f}  {m_in['rmse']:8.4f}  {m_in['mae']:8.4f}  {m_in['n']:10,}")
    print(f"{'Outside mask (e ≥ τ)':30s} {m_out['absrel']:8.4f}  {m_out['rmse']:8.4f}  {m_out['mae']:8.4f}  {m_out['n']:10,}")

    pct_absrel = 100 * (m_out['absrel'] - m_in['absrel']) / m_out['absrel']
    pct_rmse   = 100 * (m_out['rmse']   - m_in['rmse'])   / m_out['rmse']
    print(f"\nDepth error inside mask is {pct_absrel:.1f}% lower (AbsRel), {pct_rmse:.1f}% lower (RMSE) than outside.")

    # ------------------------------------------------------------------
    # 3. Binned correlation: all available thresholds
    #    photometric error bin centre  vs.  mean depth error in that bin
    # ------------------------------------------------------------------
    print(f"\n=== Binned Correlation: Photometric Error → Depth Error ===")

    all_mask_dirs = sorted(seq_dir.glob("photo_masks_rgbonly_low*_sampleevery2"))
    bin_thresholds = []
    for d in all_mask_dirs:
        tag = d.name.replace("photo_masks_rgbonly_low", "").replace("_sampleevery2", "")
        try:
            t = float(tag) / (10 ** (len(tag) - 1))
            bin_thresholds.append((t, d))
        except ValueError:
            pass
    bin_thresholds.sort()

    # Build a single pixel-level "bin index" from the cumulative masks.
    # Bin i: pixel has e ∈ [thresh[i-1], thresh[i]).
    # We need all masks to share the same frame set — use the first mask dir's files.
    ref_files = sorted(bin_thresholds[0][1].glob("*.png"))

    all_bin_da2, all_bin_gt, all_bin_thresholds_px = [], [], []
    for rf in ref_files:
        stem = rf.stem
        da2_path = seq_dir / "depths_da2" / f"{stem}.png"
        gt_path  = seq_dir / "depths_gt"  / f"{stem}.png"
        if not da2_path.exists() or not gt_path.exists():
            continue
        da2 = read_kitti_depth(da2_path)
        gt  = read_kitti_depth(gt_path)
        da2c = np.clip(da2, 0, args.max_depth)

        # Load cumulative masks (low{t} = True where e < t)
        cum_masks = []
        for t, mdir in bin_thresholds:
            mpath = mdir / f"{stem}.png"
            if mpath.exists():
                m = read_mask(mpath)
                if m.shape != gt.shape:
                    m = cv2.resize(m.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                m = np.zeros(gt.shape, dtype=bool)
            cum_masks.append(m)

        # Assign bin index: first threshold bin the pixel falls into
        bin_idx = np.full(gt.shape, len(bin_thresholds), dtype=np.int32)
        for i, m in enumerate(cum_masks):
            bin_idx = np.where(m & (bin_idx == len(bin_thresholds)), i, bin_idx)

        valid = np.isfinite(gt) & np.isfinite(da2c) & (gt > 0.5) & (gt < args.max_depth)
        all_bin_da2.append(da2c[valid])
        all_bin_gt.append(gt[valid])
        all_bin_thresholds_px.append(bin_idx[valid])

    all_bin_da2 = np.concatenate(all_bin_da2)
    all_bin_gt  = np.concatenate(all_bin_gt)
    all_bin_idx = np.concatenate(all_bin_thresholds_px)

    # Bin centres: use threshold midpoints
    thresholds_only = [t for t, _ in bin_thresholds]
    bin_labels = []
    bin_centres = []
    bin_absrel  = []
    bin_rmse    = []
    bin_counts  = []

    edges = [0.0] + thresholds_only[:-1] + [thresholds_only[-1]]
    for i in range(len(bin_thresholds)):
        t_lo = thresholds_only[i - 1] if i > 0 else 0.0
        t_hi = thresholds_only[i]
        sel = all_bin_idx == i
        if sel.sum() < 10:
            continue
        err = np.abs(all_bin_da2[sel] - all_bin_gt[sel])
        centre = (t_lo + t_hi) / 2
        bin_labels.append(f"[{t_lo:.2f}, {t_hi:.2f})")
        bin_centres.append(centre)
        bin_absrel.append(float(np.mean(err / all_bin_gt[sel])))
        bin_rmse.append(float(np.sqrt(np.mean((all_bin_da2[sel] - all_bin_gt[sel]) ** 2))))
        bin_counts.append(int(sel.sum()))

    # Pixels in the last bin (e >= highest threshold)
    sel_last = all_bin_idx == len(bin_thresholds)
    if sel_last.sum() >= 10:
        t_lo = thresholds_only[-1]
        err  = np.abs(all_bin_da2[sel_last] - all_bin_gt[sel_last])
        bin_labels.append(f"[{t_lo:.2f}, ∞)")
        bin_centres.append(t_lo + 0.05)
        bin_absrel.append(float(np.mean(err / all_bin_gt[sel_last])))
        bin_rmse.append(float(np.sqrt(np.mean((all_bin_da2[sel_last] - all_bin_gt[sel_last]) ** 2))))
        bin_counts.append(int(sel_last.sum()))

    print(f"\n{'Bin (photo error)':22s}  {'AbsRel':>8}  {'RMSE':>8}  {'N pixels':>10}")
    for label, absrel, rmse, n in zip(bin_labels, bin_absrel, bin_rmse, bin_counts):
        print(f"{label:22s}  {absrel:8.4f}  {rmse:8.4f}  {n:10,}")

    # Pearson & Spearman correlations (bin-level)
    if len(bin_centres) >= 3:
        r_pearson, p_pearson   = stats.pearsonr(bin_centres, bin_absrel)
        r_spearman, p_spearman = stats.spearmanr(bin_centres, bin_absrel)
        print(f"\nBin-level Pearson  r={r_pearson:.4f}  p={p_pearson:.4e}  (AbsRel vs photo-error bin centre)")
        print(f"Bin-level Spearman r={r_spearman:.4f}  p={p_spearman:.4e}")

    # Also compute per-pixel (continuous) correlation using bin index as proxy
    # (bin_idx is ordinal, so Spearman is more appropriate)
    depth_err_all = np.abs(all_bin_da2 - all_bin_gt)
    absrel_all    = depth_err_all / all_bin_gt
    # Subsample for speed if large
    if len(all_bin_idx) > 500_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_bin_idx), 500_000, replace=False)
        sx, sy = all_bin_idx[idx].astype(float), absrel_all[idx]
    else:
        sx, sy = all_bin_idx.astype(float), absrel_all

    r_pix_s, p_pix_s = stats.spearmanr(sx, sy)
    r_pix_p, p_pix_p = stats.pearsonr(sx, sy)
    print(f"\nPer-pixel (bin-ordinal proxy):")
    print(f"  Spearman r={r_pix_s:.4f}  p={p_pix_s:.4e}")
    print(f"  Pearson  r={r_pix_p:.4f}  p={p_pix_p:.4e}")

    # ------------------------------------------------------------------
    # 4. Summary for rebuttal table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("REBUTTAL TABLE (Table R1) — fill in these numbers:")
    print(f"{'='*60}")
    print(f"  Mask threshold used: τ = {chosen_thresh}")
    print(f"  (Note: exact τ=0.18 mask {'found' if exact_mask_dir.exists() else 'NOT found locally; used τ='+str(chosen_thresh)+' as proxy'})")
    print()
    print(f"  Inside mask  — AbsRel: {m_in['absrel']:.4f}   RMSE: {m_in['rmse']:.4f}   N: {m_in['n']:,}")
    print(f"  Outside mask — AbsRel: {m_out['absrel']:.4f}   RMSE: {m_out['rmse']:.4f}   N: {m_out['n']:,}")
    print()
    print(f"  Relative reduction (inside vs outside):")
    print(f"    AbsRel: {pct_absrel:.1f}%    RMSE: {pct_rmse:.1f}%")
    if len(bin_centres) >= 3:
        print()
        print(f"  Binned Pearson  r = {r_pearson:.3f}   p = {p_pearson:.2e}")
        print(f"  Binned Spearman r = {r_spearman:.3f}   p = {p_spearman:.2e}")
        print(f"  Per-pixel Spearman r = {r_pix_s:.3f}   p = {p_pix_s:.2e}")


if __name__ == "__main__":
    main()
