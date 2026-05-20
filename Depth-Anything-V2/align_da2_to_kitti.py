import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def read_kitti_depth_png(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)

    depth = raw.astype(np.float32) / 256.0
    depth[raw < 2] = -1.0
    return depth


def fit_scale_shift(pred, gt, mask):
    x = pred[mask].astype(np.float32).reshape(-1)
    y = gt[mask].astype(np.float32).reshape(-1)

    if x.shape[0] < 100:
        return None, None

    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    return float(sol[0]), float(sol[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--da2-npy-dir", required=True)
    parser.add_argument("--gt-depth-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--min-depth", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npy_files = sorted([f for f in os.listdir(args.da2_npy_dir) if f.endswith(".npy")])

    errors = []

    for f in tqdm(npy_files):
        stem = os.path.splitext(f)[0]

        pred = np.load(os.path.join(args.da2_npy_dir, f)).astype(np.float32)
        gt_path = os.path.join(args.gt_depth_dir, stem + ".png")
        gt = read_kitti_depth_png(gt_path)

        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)

        mask = (
            np.isfinite(pred)
            & np.isfinite(gt)
            & (gt > args.min_depth)
            & (gt < args.max_depth)
        )

        a, b = fit_scale_shift(pred, gt, mask)

        if a is None:
            print(f"Skip {stem}: too few valid GT depth points")
            continue

        aligned = a * pred + b
        aligned = np.clip(aligned, 0.0, args.max_depth)

        # Invalid region remains 0 in saved png.
        aligned_u16 = (aligned * 256.0).astype(np.uint16)
        out_path = os.path.join(args.out_dir, stem + ".png")
        cv2.imwrite(out_path, aligned_u16)

        abs_err = np.mean(np.abs(aligned[mask] - gt[mask]))
        errors.append(abs_err)

    print(f"Done. Saved to {args.out_dir}")
    if len(errors) > 0:
        print(f"Mean alignment Abs Error on valid LiDAR pixels: {np.mean(errors):.4f} m")


if __name__ == "__main__":
    main()
