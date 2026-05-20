import os
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--checkpoint", default="checkpoints/depth_anything_v2_vits.pth")
    parser.add_argument("--input-size", type=int, default=518)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    model = DepthAnythingV2(**model_configs[args.encoder])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model = model.to(device).eval()

    os.makedirs(args.out_dir, exist_ok=True)

    image_names = [
        f for f in sorted(os.listdir(args.img_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Device: {device}")
    print(f"Found {len(image_names)} images.")

    for name in tqdm(image_names):
        img_path = os.path.join(args.img_dir, name)
        raw_img = cv2.imread(img_path)

        if raw_img is None:
            print(f"Failed to read {img_path}")
            continue

        with torch.no_grad():
            depth = model.infer_image(raw_img, args.input_size)

        stem = os.path.splitext(name)[0]
        np.save(os.path.join(args.out_dir, stem + ".npy"), depth.astype(np.float32))

    print("Done.")


if __name__ == "__main__":
    main()
