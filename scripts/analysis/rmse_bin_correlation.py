"""Compute RMSE-based binned correlation and median GT depth per bin."""
import sys
sys.path.insert(0, "/Users/waynechu/miniconda3/envs/cs231n-final/lib/python3.10/site-packages")

import cv2
import numpy as np
from pathlib import Path
from scipy import stats
from PIL import Image

SEQ = Path("/Users/waynechu/cs231n-final-project/data/kitti/kitti_select_static_5seq/KITTISeq02_2011_10_03_drive_0034_sync_llffdtu_s2749_e2929_densegt")

def read_kitti(p):
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    d = raw.astype(np.float32) / 256.0
    d[raw == 0] = np.nan
    return d

thresholds_dirs = sorted(SEQ.glob("photo_masks_rgbonly_low*_sampleevery2"))
thresh_vals = []
for d in thresholds_dirs:
    tag = d.name.replace("photo_masks_rgbonly_low","").replace("_sampleevery2","")
    thresh_vals.append((float(tag)/10**(len(tag)-1), d))
thresh_vals.sort()
print("Thresholds:", [t for t,_ in thresh_vals])

# Use intersection of stems across all mask directories
stem_sets = [set(p.stem for p in d.glob("*.png")) for _, d in thresh_vals]
common_stems = sorted(set.intersection(*stem_sets))
print(f"Frames in all mask dirs: {len(common_stems)}")

all_da2, all_gt, all_bidx = [], [], []
for stem in common_stems:
    da2 = read_kitti(SEQ/"depths_da2"/f"{stem}.png")
    gt  = read_kitti(SEQ/"depths_gt"/f"{stem}.png")
    da2c = np.clip(da2, 0, 80)
    valid = np.isfinite(gt) & np.isfinite(da2c) & (gt>0.5) & (gt<80)
    masks = []
    for t,mdir in thresh_vals:
        mp = mdir/f"{stem}.png"
        m = np.array(Image.open(mp))>127
        if m.shape != gt.shape:
            m = cv2.resize(m.astype(np.uint8),(gt.shape[1],gt.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        masks.append(m)
    bidx = np.full(gt.shape, len(thresh_vals), dtype=np.int32)
    for i,m in enumerate(masks):
        bidx = np.where(m & (bidx==len(thresh_vals)), i, bidx)
    all_da2.append(da2c[valid])
    all_gt.append(gt[valid])
    all_bidx.append(bidx[valid])

da2f = np.concatenate(all_da2)
gtf  = np.concatenate(all_gt)
bidx = np.concatenate(all_bidx)

tonly = [t for t,_ in thresh_vals]
print()
print(f"{'Bin':22s}  {'AbsRel':>8}  {'RMSE':>8}  {'Median_GT(m)':>12}  {'N':>10}")
rmse_v, absrel_v, cen_v = [], [], []
for i in range(len(thresh_vals)):
    t_lo = tonly[i-1] if i>0 else 0.0
    t_hi = tonly[i]
    sel = bidx==i
    if sel.sum()<10: continue
    err = np.abs(da2f[sel]-gtf[sel])
    ar = float(np.mean(err/gtf[sel]))
    rm = float(np.sqrt(np.mean((da2f[sel]-gtf[sel])**2)))
    md = float(np.median(gtf[sel]))
    c = (t_lo+t_hi)/2
    print(f"[{t_lo:.2f},{t_hi:.2f})  {ar:>8.4f}  {rm:>8.4f}  {md:>12.2f}  {sel.sum():>10,}")
    rmse_v.append(rm); absrel_v.append(ar); cen_v.append(c)

sel = bidx==len(thresh_vals)
if sel.sum()>10:
    err = np.abs(da2f[sel]-gtf[sel])
    ar = float(np.mean(err/gtf[sel]))
    rm = float(np.sqrt(np.mean((da2f[sel]-gtf[sel])**2)))
    md = float(np.median(gtf[sel]))
    print(f"[{tonly[-1]:.2f},inf)   {ar:>8.4f}  {rm:>8.4f}  {md:>12.2f}  {sel.sum():>10,}")
    rmse_v.append(rm); absrel_v.append(ar); cen_v.append(tonly[-1]+0.05)

print()
rp_r, pp_r = stats.pearsonr(cen_v, rmse_v)
rs_r, ps_r = stats.spearmanr(cen_v, rmse_v)
rp_a, pp_a = stats.pearsonr(cen_v, absrel_v)
rs_a, ps_a = stats.spearmanr(cen_v, absrel_v)
print(f"RMSE   Pearson  r={rp_r:.4f}  p={pp_r:.4e}")
print(f"RMSE   Spearman r={rs_r:.4f}  p={ps_r:.4e}")
print(f"AbsRel Pearson  r={rp_a:.4f}  p={pp_a:.4e}")
print(f"AbsRel Spearman r={rs_a:.4f}  p={ps_a:.4e}")
