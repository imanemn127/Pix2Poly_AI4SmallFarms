"""
Compute per-channel mean and std of the training patches (32x32 Sentinel-2).
Reads bands [1, 2, 3] (indices 0,1,2 in rasterio = first 3 bands of the GeoTIFF).
Values are in raw uint16 reflectance (0-10000).

Usage:
    conda run -n ai4sf python scripts/compute_normalization.py
    conda run -n ai4sf python scripts/compute_normalization.py --max_images 2000
"""

import argparse
import json
import os
import numpy as np
import rasterio
from tqdm import tqdm

DATA_ROOT  = "/home/imane/DATA/AI4SmallFarms/sentinel-2-asia"
COCO_JSON  = os.path.join(DATA_ROOT, "output_coco_32/train_coco.json")
MAX_PIXEL  = 10000.0   # normalization divisor used during training


def main(max_images=None):
    with open(COCO_JSON) as f:
        images = json.load(f)["images"]

    if max_images:
        rng = np.random.default_rng(0)
        images = rng.choice(images, size=min(max_images, len(images)), replace=False).tolist()

    print(f"Computing stats on {len(images)} patches …")

    # Welford online algorithm — avoids loading all pixels into RAM
    n      = np.zeros(3, dtype=np.float64)
    mean   = np.zeros(3, dtype=np.float64)
    M2     = np.zeros(3, dtype=np.float64)

    for img_info in tqdm(images):
        path = os.path.join(DATA_ROOT, img_info["image_path"])
        with rasterio.open(path) as src:
            data = src.read([1, 2, 3]).astype(np.float64) / MAX_PIXEL  # (3, H, W)

        for c in range(3):
            pixels = data[c].ravel()
            for x in pixels:
                n[c]    += 1
                delta    = x - mean[c]
                mean[c] += delta / n[c]
                M2[c]   += delta * (x - mean[c])

    std = np.sqrt(M2 / (n - 1))

    print("\n=== Results (values after dividing by max_pixel_value=10000) ===")
    print(f"image_mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"image_std:  [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    print("\nPaste these into config/encoder/vit_s2.yaml:")
    print(f"  image_mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  image_std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print(f"  image_max_pixel_value: {MAX_PIXEL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_images", type=int, default=None,
                        help="Use a random subset (seed=0) for speed. Default: all images.")
    args = parser.parse_args()
    main(args.max_images)
